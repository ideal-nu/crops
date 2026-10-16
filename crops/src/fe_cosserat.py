"""Finite element implementation of the Cosserat rod.

Implements a geometrically nonlinear static equilibrium solver for Cosserat rods with
support for gel swelling, hard-magnetic actuation, wall contact, and self-contact.
"""

from typing import Optional, Tuple

import jax
import jax.experimental.sparse as jax_sprs
import jax.numpy as jnp
import jax_dataclasses as jxd
import numpy as np

import crops.src.bc as _bc
import crops.src.contact as _contact
import crops.src.material as _material
import crops.src.mesher as _mesher
import crops.src.solver as _solver
import crops.src.utils as _utils

_Strn = _mesher.StrainComponent
_Curv = _mesher.CurvatureComponent


@jxd.pytree_dataclass
class FlorySolvent:
  """Parameters for the Flory-Rehner solvent for swelling of gels.

  Attributes:
    ref_chem_potential: Scalar float μ⁰ [J/mol]. Reference chemical potential. Usually
      taken as zero.
    bath_temperature: Scalar float [K]. Absolute temperature of the solvent bath. A
      higher temperature increases the swelling driving force.
    molar_volume: Scalar float Ω [m³/mol]. Molar volume of the solvent. Used to
      compute the elastic resistance to swelling in the polymer network. A larger molar
      volume means the solvent molecules are larger and harder for it to penetrate the
      polymer network, leading to more elastic resistance and less swelling.
  """

  ref_chem_potential: float
  bath_temperature: float
  molar_volume: float


def _compute_fiber_polymer_volume_fraction(
  total_fiber_axial_stretch: jnp.ndarray,
  flory_parameter: float,
  shear_modulus: float,
  chem_potential: float,
  solvent: FlorySolvent,
  bisection_settings: dict,
) -> jnp.ndarray:
  """Compute the polymer volume fraction φ for a 1D rod gel fiber via bisection.

  For details see Hong, Wei, Zishun Liu, and Zhigang Suo. "Inhomogeneous swelling
  of a gel in equilibrium with a solvent and mechanical load." International Journal
  of Solids and Structures 46, no. 17 (2009): 3282-3289.

  Solves the Flory-Rehner implicit equation for φ:

      (μ⁰ − μ) / (RT) + ln(1 − φ) + φ + χφ² + (Ω·G/RT)·φ·(λ_t² − 1) = 0

  where λ_t² = 1/(φ·λ_axial) for a 1D rod fiber with two free transverse
  directions. The bisection is wrapped with a custom JVP (implicit function
  theorem rule) so that dφ/d(inputs) propagates correctly through `jax.grad`
  without unrolling the bisection iterations.

  Args:
    total_fiber_axial_stretch: Scalar float λ_axial,ₖ. Total fiber axial stretch
      from the dry natural reference. Equal to 1.0 at the dry reference state.
    flory_parameter: Scalar float χₖ ≥ 0. Flory-Huggins interaction parameter.
    shear_modulus: Scalar float Gₖ [Pa]. Dry-state neo-Hookean shear modulus.
    chem_potential: Scalar float μ [J/mol]. Current bath chemical potential.
    solvent: `FlorySolvent` or `None`. Solvent parameters (μ⁰, T, Ω) for the
      Flory-Rehner swelling model.
    bisection_settings: Settings for the bisection solver with keys `lower` (float,
      lower bound for φ), `upper` (float, upper bound for φ), `tol` (float, interval
      width tolerance), `max_iter` (int, maximum iterations).

  Returns:
    Scalar float φ ∈ (lower, upper). Polymer volume fraction at chemical equilibrium.
      φ ≈ 1.0 is the dry state; φ → 0.0 is the fully swollen limit.
  """

  def _implicit_fn(
    phi: jnp.ndarray,
    flory_parameter: float,
    shear_modulus: float,
    chem_potential: float,
    solvent: FlorySolvent,
    total_fiber_axial_stretch: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes the Flory-Rehner implicit residual for the 1D rod fiber."""
    rtheta = _utils.GAS_CONSTANT * solvent.bath_temperature
    chemical_force = (solvent.ref_chem_potential - chem_potential) / rtheta
    mixing_energy = _utils.safe_log(1.0 - phi) + phi + flory_parameter * phi**2
    # 1D fiber: both transverse dirs free → λ_t² = 1/(φ·λ_axial)
    lambda_t_sq = _utils.safe_divide(1.0, phi * total_fiber_axial_stretch)
    elastic_energy = (
      (solvent.molar_volume * shear_modulus / rtheta) * phi * (lambda_t_sq - 1.0)
    )
    return chemical_force + mixing_energy + elastic_energy

  return _solver.solve_bisection(
    _implicit_fn,
    bisection_settings,
    flory_parameter,
    shear_modulus,
    chem_potential,
    solvent,
    total_fiber_axial_stretch,
  )


def _compute_elem_kinematics(
  elem_node_dofs: jnp.ndarray,
  elem_ref_arc_length: jnp.ndarray,
  ref_pos_pair: jnp.ndarray,
  directors_0_pair: jnp.ndarray,
) -> Tuple[_mesher.CosseratKinematics, jnp.ndarray]:
  """Reconstruct current element kinematics from the nodal DOFs.

  Maps the relative dof vector for one element to the Cosserat strain [η₁, η₂, ν] and
  curvature [κ₁, κ₂, κ₃] measures, together with the geodesic mean director frame
  required for frame conversions downstream.

  All DOF entries are increments from the stress-free reference configuration.
  Translational entries Δr = r_current − r_ref [m]; rotational entries are Rodrigues
  vectors δθ ∈ ℝ³ [rad] that rotate the reference directors via R(δθ). At zero dofs
  the function returns the reference kinematics and the reference mean directors.

  Per-node DOF block layout (6 dofs per node, 12 per element):

    node i:        0     1     2      3      4     5
                 [Δxᵢ,  Δyᵢ,  Δzᵢ,  δθₓᵢ,  δθᵧᵢ,  δθᵤᵢ,
    node i+1:     6           7        8          9           10          11  ]
                 Δx_{i+1}, Δy_{i+1}, Δz_{i+1}, δθₓ_{i+1}, δθᵧ_{i+1}, δθᵤ_{i+1}]

  Args:
    elem_node_dofs: Array of shape `(num_dofs_per_elem=12,)`. All entries are zero at
      the reference configuration. Translational entries [m]; rotational entries [rad].
    elem_ref_arc_length: Scalar float Δŝₑ [m]. Reference arc-length span of this elem.
    ref_pos_pair: Array of shape `(num_nodes_per_elem=2, num_dim=3)` [m]. Reference
      (undeformed) positions of the two bounding nodes in global frame of reference.
    directors_0_pair: Array of shape `(num_nodes_per_elem=2, num_dirs=3, num_dim=3)`.
      Reference director frames at the two bounding nodes. `directors_0_pair[n, k, :]`
      is the k-th material director dₖ at node n (k=0 → d₁, k=1 → d₂, k=2 → d₃). Rows
      are orthonormal, det = +1.

  Returns:
    A tuple containing:
      - `CosseratKinematics` with `strain` = [η₁, η₂, ν] and `curvature` =
        [κ₁, κ₂, κ₃], each of shape `(num_components=3,)`.
      - `directors_mean`: Array of shape `(num_dirs=3, num_dim=3)`. Geodesic
        SO(3) midpoint of the two current element-node director frames at
        t = 0.5.
  """

  pos_i = ref_pos_pair[0] + elem_node_dofs[0:3]
  theta_inc_i = elem_node_dofs[3:6]

  pos_ip1 = ref_pos_pair[1] + elem_node_dofs[6:9]
  theta_inc_ip1 = elem_node_dofs[9:12]

  directors_i = _utils.rotate_directors(
    rotation=theta_inc_i,
    ref_directors=directors_0_pair[0],
  )
  directors_ip1 = _utils.rotate_directors(
    rotation=theta_inc_ip1,
    ref_directors=directors_0_pair[1],
  )
  directors_mean = _utils.interpolate_directors_geodesic_so3(
    directors_i=directors_i,
    directors_ip1=directors_ip1,
    t=0.5,
  )
  kin = _mesher.compute_discrete_cosserat_kinematics(
    node_coords=jnp.stack([pos_i, pos_ip1]),
    directors=jnp.stack([directors_i, directors_ip1]),
    arc_coords=jnp.array([0.0, elem_ref_arc_length]),
  )
  return (
    _mesher.CosseratKinematics(
      strain=kin.strain[0],
      curvature=kin.curvature[0],
    ),
    directors_mean,
  )


def _project_rod_kinematics_onto_fiber(
  kinem: _mesher.CosseratKinematics,
  fiber_offset: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Project rod-level Cosserat kinematics onto the local axes of one fiber.

  A fiber at cross-section offset (y₁, y₂) from the rod centerline experiences
  different axial stretch and shear strains than the centerline itself, due to bending
  (which stretches off-axis fibers) and twist (which couples axial and lateral positions
  into shear). This function computes the total fiber kinematic projections.

  The fiber centroid position is r_f(ŝ) = r(ŝ) + y₁ d₁(ŝ) + y₂ d₂(ŝ).
  Differentiating along the arc and using the Cosserat frame transport identities

      ∂d₁/∂ŝ = κ × d₁ = κ₃ d₂ − κ₂ d₃,
      ∂d₂/∂ŝ = κ × d₂ = −κ₃ d₁ + κ₁ d₃,

      with, ∂r/∂ŝ  = η₁ d₁ + η₂ d₂ + ν d₃

      ∂r_f/∂ŝ = ∂r/∂ŝ + y₁ (κ₃ d₂ − κ₂ d₃) + y₂ (−κ₃ d₁ + κ₁ d₃)

  the projections of ∂r_f/∂ŝ onto the material frame directors are:

      λ_fiber = (∂r_f/∂ŝ) · d₃ = ν − y₁ κ₂ + y₂ κ₁
      γ₁_fiber = (∂r_f/∂ŝ) · d₁ = η₁ − y₂ κ₃
      γ₂_fiber = (∂r_f/∂ŝ) · d₂ = η₂ + y₁ κ₃

  Args:
    kinem: `CosseratKinematics` for any configuration (current or reference).
    fiber_offset: Array of shape `(num_crossec_dim=2,)` [m]. Material-frame offsets
      (y₁, y₂) of the fiber centroid along the current d₁ and d₂ directors from
      the rod centerline.

  Returns:
    A tuple `(lambda_fiber, gamma1_fiber, gamma2_fiber)`. All three are scalar floats
      representing the total axial stretch, d₁-shear, and d₂-shear of the fiber at this
      configuration.
  """
  strain, curv = kinem.strain, kinem.curvature
  y1, y2 = fiber_offset[0], fiber_offset[1]

  lambda_fiber = (
    strain[_Strn.AXIAL_STRETCH]
    - y1 * curv[_Curv.BENDING_D2]
    + y2 * curv[_Curv.BENDING_D1]
  )
  gamma1_fiber = strain[_Strn.SHEAR_D1] - y2 * curv[_Curv.TWIST]
  gamma2_fiber = strain[_Strn.SHEAR_D2] + y1 * curv[_Curv.TWIST]

  return lambda_fiber, gamma1_fiber, gamma2_fiber


def _compute_fiber_kinematics(
  elem_kinematics: _mesher.CosseratKinematics,
  fiber_offset: jnp.ndarray,
  elem_ref_kinematics: _mesher.CosseratKinematics,
  eps: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Extract elastic kinematic invariants for a single off-axis fiber.

      λ₀        = λ_fiber_ref      (clamped ≥ eps)
      λ_axial   = λ_fiber_cur / λ₀
      η₁_eff    = (γ₁_fiber_cur − γ₁_fiber_ref) / λ₀
      η₂_eff    = (γ₂_fiber_cur − γ₂_fiber_ref) / λ₀

  All three return values equal their reference values (1, 0, 0) at the ref config.

  Args:
    elem_kinematics: `CosseratKinematics` for an element at the current deformed state.
    fiber_offset: Array of shape `(num_crossec_dim=2,)` [m]. Material-frame offsets
      (y₁, y₂) of the fiber centroid along the current d₁ and d₂ directors from the
      rod centerline. Invariant under deformation.
    elem_ref_kinematics: `CosseratKinematics` at the stress-free reference config.
    eps: Scalar float. Floor applied to λ₀ to prevent division by zero in degenerate
      reference configurations. Default 1e-12.

  Returns:
    A tuple `(lambda_axial, eta1_eff, eta2_eff)`. All three are scalar floats. At the
      reference configuration: lambda_axial = 1.0, eta1_eff = 0.0,  eta2_eff = 0.0.

  """
  lambda_cur, gamma1_cur, gamma2_cur = _project_rod_kinematics_onto_fiber(
    kinem=elem_kinematics,
    fiber_offset=fiber_offset,
  )
  lambda_ref, gamma1_ref, gamma2_ref = _project_rod_kinematics_onto_fiber(
    kinem=elem_ref_kinematics,
    fiber_offset=fiber_offset,
  )

  ref_fiber_stretch = jnp.maximum(lambda_ref, eps)
  lambda_axial = lambda_cur / ref_fiber_stretch
  eta1_eff = (gamma1_cur - gamma1_ref) / ref_fiber_stretch
  eta2_eff = (gamma2_cur - gamma2_ref) / ref_fiber_stretch

  return lambda_axial, eta1_eff, eta2_eff


def _compute_fiber_deformation_gradient(
  elem_kinematics: _mesher.CosseratKinematics,
  fiber_offset: jnp.ndarray,
  polymer_vol_frac: float,
  elem_ref_kinematics: _mesher.CosseratKinematics,
  eps: float = 1e-12,
) -> jnp.ndarray:
  """Assemble the deformation gradient F in the rod material frame.

  Constructs the deformation gradient F in the element material frame {d₁, d₂, d₃}
  from the elastic fiber kinematics and assumes incompressibility J = det(F) = 1/φ.

      λ_t = 1 / √(φ · λ_axial)          (isotropic transverse contraction)

                ⎡λ_t    0      η₁_eff ⎤
           F  = ⎢ 0    λ_t     η₂_eff ⎥
                ⎣ 0     0      λ_axial ⎦

  The multiplicative decomposition F = Fₑ · Fₛ · F₀ is applied implicitly. F₀ removes
  the reference geometry, Fₛ encodes isotropic swelling, and Fₑ is the elastic
  incompressible part. For dry elastic rods (φ = 1) λ_t = 1/√λ_axial and det(F) = 1.

  Args:
    elem_kinematics: `CosseratKinematics` for element at the current deformed state.
    fiber_offset: Array of shape `(num_crossec_dim=2,)` [m]. Material-frame offsets
      (y₁, y₂) of the fiber centroid along the current d₁ and d₂ directors from the rod
      centerline.
    polymer_vol_frac: Scalar float φ ∈ (0, 1]. Polymer volume fraction. Pass 1.0 for dry
      rods. For swelling rods, pass the φ returned computing the polymer volume fraction
      as dictated by the Flory-Rehner model.
    elem_ref_kinematics: `CosseratKinematics` at the stress-free reference config.
      Provides the reference fiber projections for the F₀ decomposition.
    eps: Scalar float. Floor for axial fiber stretch and reference stretch.

  Returns:
    Array of shape `(num_dim=3, num_dim=3)`. Deformation gradient in the material frame
    {d₁, d₂, d₃}. Upper-triangular with det(F) = 1/φ by construction from the
    incompressibility requirement.
  """
  lambda_axial, eta1_eff, eta2_eff = _compute_fiber_kinematics(
    elem_kinematics=elem_kinematics,
    fiber_offset=fiber_offset,
    elem_ref_kinematics=elem_ref_kinematics,
    eps=eps,
  )
  lambda_t = jnp.sqrt(1.0 / (polymer_vol_frac * lambda_axial))
  return jnp.array(
    [
      [lambda_t, 0.0, eta1_eff],
      [0.0, lambda_t, eta2_eff],
      [0.0, 0.0, lambda_axial],
    ]
  )


def _def_grad_material_to_global(
  def_grad_material: jnp.ndarray,
  directors_current: jnp.ndarray,
  directors_ref: jnp.ndarray,
) -> jnp.ndarray:
  """Convert the material-frame deformation gradient to the global Cartesian frame.

  `_compute_fiber_deformation_gradient` assembles F in the element material frame
  {d₁, d₂, d₃}, where component F[α, β] represents the stretch/shear from reference
  material axis β to current material axis α. The global Cartesian deformation gradient
  is obtained via the two director rotation matrices:

      F_global = Q_current · F_material · Q_ref^T

  where Q ∈ SO(3) maps material-frame coordinates to global Cartesian, with
  columns equal to the directors. Using the stored row convention
  directors[k, i] = i-th global component of dₖ, so Q = directors^T:

      F_global[i, j] = Σ_{k,l} directors_current[k, i]
                       · F_material[k, l]
                       · directors_ref[l, j]

  This is needed whenever a constitutive law requires F expressed in a fixed lab
  frame rather than the co-rotating material frame. The determinant is preserved:
  det(F_global) = det(F_material) since det(Q) = 1.

  Args:
    def_grad_material: Array of shape `(num_dim=3, num_dim=3)`. Deformation gradient in
      the element material frame {d₁, d₂, d₃}, as returned by
      `_compute_fiber_deformation_gradient`.
    directors_current: Array of shape `(num_dirs=3, num_dim=3)`. Current element
      mean director frame. Rows are orthonormal (rows = d₁, d₂, d₃) with  det = +1.
    directors_ref: Array of shape `(num_dirs=3, num_dim=3)`. Reference element mean
      director frame. Computed as the geodesic SO(3) midpoint of the two reference node
      frames via `interpolate_directors_geodesic_so3` at t = 0.5.

  Returns:
    Array of shape `(num_dim=3, num_dim=3)`. Deformation gradient in the global
    Cartesian frame. Satisfies det(F_global) = det(F_material) since det(Q) = 1.
  """
  return jnp.einsum(
    "ki, kl, lj -> ij", directors_current, def_grad_material, directors_ref
  )


def _compute_elem_energy(
  elem_node_dofs: jnp.ndarray,
  elem_ref_arc_length: jnp.ndarray,
  ref_pos_pair: jnp.ndarray,
  directors_0_pair: jnp.ndarray,
  elem_ref_kinematics: _mesher.CosseratKinematics,
  fiber_offsets: jnp.ndarray,
  fiber_areas: jnp.ndarray,
  fiber_shear_moduli: jnp.ndarray,
  fiber_winding_directions: jnp.ndarray,
  fiber_reinforcement_moduli: Optional[jnp.ndarray] = None,
  fiber_remnant_mag_flux: Optional[jnp.ndarray] = None,
  fiber_flory_params: Optional[jnp.ndarray] = None,
  fiber_applied_mag_flux: Optional[jnp.ndarray] = None,
  chem_potential: Optional[float] = None,
  solvent: Optional[FlorySolvent] = None,
  bisection_settings: Optional[dict] = None,
  eps: float = 1e-12,
) -> jnp.ndarray:
  """Compute the total internal strain energy for one 3D Cosserat rod element.

  Integrates the Helmholtz free energy density over the cross-section fiber bundle via a
  quadrature rule that weights each fiber by its reference cross-section area and the
  element arc-length:

                        Uⱼ = Δŝⱼ · Σₖ ψₖ(Fₖ, Gₖ, φₖ) · Aₖ

  The per-fiber pipeline for fiber k at offset (y₁ₖ, y₂ₖ) is:

    1. Compute the element Cosserat kinematics [η₁, η₂, ν, κ₁, κ₂, κ₃] and
        the geodesic mean director frame.
    2. When swelling is active, solve the Flory-Rehner equation to obtain the amount of
        swelling. For dry rods, φₖ = 1.
    3. Assemble the material-frame deformation gradient Fₖ passing φₖ so that the
       transverse stretch λ_t and det(F) = 1/φₖ are consistent.
    4. Evaluate the Flory-Rehner elastic energy.  At φ = 1 this reduces to the
       incompressible Neo-Hookean G/2(I₁ − 3).
    5. When magnetic actuation is active, rotate Fₖ to the global Cartesian
       frame and add the hard-magnetic Helmholtz energy.


  Optional physics are enabled by supplying the corresponding non-None
  arguments. Any combination of swelling and magnetic actuation is supported.

  Args:
    elem_node_dofs: Array of shape `(num_dofs_per_elem=12,)`. Relative nodal dofs for
      the two bounding nodes. All zero at the reference configuration. Translational
      entries [m], rotational entries [rad].
    elem_ref_arc_length: Scalar float Δŝⱼ [m]. Reference element arc-length. Serves as
      the integration weight in Uⱼ = Δŝⱼ · Σₖ ψₖ Aₖ.
    ref_pos_pair: Array of shape `(num_nodes_per_elem=2, num_dim=3)` [m]. Reference
      centroid positions of the two bounding nodes in the global Cartesian frame.
    directors_0_pair: Array of shape `(num_nodes_per_elem=2, num_dirs=3, num_dim=3)`.
      Ref. director frames at the two bounding nodes. Rows are orthonormal, det = +1.
    elem_ref_kinematics: `CosseratKinematics` at the stress-free reference config.
      for this element. Provides reference strains and curvatures for the multiplicative
      F₀ decomposition and the bisection's reference fiber stretch.
    fiber_offsets: Array of shape `(num_fibers, num_crossec_dim=2)` [m]. Material-frame
      offsets (y₁, y₂) for each fiber along the current d₁ and d₂ directors from the
      rod centerline.
    fiber_areas: Array of shape `(num_fibers,)` [m²]. Reference cross-section area of
      each fiber. Quadrature weight in Uⱼ = Δŝⱼ · Σₖ ψₖ Aₖ.
    fiber_shear_moduli: Array of shape `(num_fibers,)` [Pa]. Dry-state Neo-Hookean
      shear modulus Gₖ per fiber.
    fiber_winding_directions: Array of shape `(num_fibers, num_dim=3)`. Unit
      reinforcement direction of each fiber, resolved on the reference material axes
      (d₁, d₂, d₃), as stored on `Fibers.winding_directions`.
    fiber_reinforcement_moduli: Array of shape `(num_fibers,)` [Pa] or `None`.
      Stiffness ηₖ ≥ 0 of the transversely isotropic reinforcement of each fiber.
      Pass `None`, or an array of zeros, for a purely isotropic rod, in which case
      the response is identical to the unreinforced model.
    fiber_remnant_mag_flux: Array of shape `(num_fibers, num_dim=3)` [T] or `None`.
      Remnant magnetic flux density Bᵣₖ per fiber in the reference global Cartesian
      frame (the lab frame at magnetic programming time). F is rotated to the global
      frame internally, so no pre-rotation is required by the caller. Must be
      accompanied by `fiber_applied_mag_flux`.
    fiber_flory_params: Array of shape `(num_fibers,)` or `None`. Flory-Huggins
      interaction parameter χₖ ≥ 0 per fiber. Values near zero give maximal
      swelling (good solvent); values above 0.5 suppress swelling (poor solvent).
      Pass `None` for dry elastic rods (φ = 1 throughout).
    fiber_applied_mag_flux: Array of shape `(num_dim=3,)` [T] or `None`. Applied ext.
      magnetic flux density B_a in the current global Cartesian lab frame, shared by all
      fibers in this element. Pass `None` for non-magnetic rods.
    chem_potential: Scalar float μ [J/mol] or `None`. Bath chemical potential at the
      current load step. Must be non-None when `fiber_flory_params` is  non-None.
    solvent: `FlorySolvent` or `None`. Solvent parameters (μ⁰, T, Ω) for the
      Flory-Rehner swelling model. Must be non-None when `fiber_flory_params` is
      non-None.
    bisection_settings: Python dict or `None`. Settings for the bisection solver
      with keys `lower` (float, lower bound for φ), `upper` (float, upper bound
      for φ), `tol` (float, interval width tolerance), `max_iter` (int, maximum
      iterations). Must be non-None when `fiber_flory_params` is non-None.
    eps: Scalar float. Floor applied to the fiber axial stretch and reference
      fiber stretch to guard against division by zero. Default 1e-12.

  Returns:
    Scalar float Uⱼ [J]. Total element strain energy. Zero at the reference
    configuration (zero DOFs, φ = 1, no applied magnetic field).
  """
  is_swelling = fiber_flory_params is not None
  is_magnetic = fiber_remnant_mag_flux is not None
  is_reinforced = fiber_reinforcement_moduli is not None

  elem_kinematics, directors_mean_current = _compute_elem_kinematics(
    elem_node_dofs=elem_node_dofs,
    elem_ref_arc_length=elem_ref_arc_length,
    ref_pos_pair=ref_pos_pair,
    directors_0_pair=directors_0_pair,
  )
  directors_mean_ref = _utils.interpolate_directors_geodesic_so3(
    directors_i=directors_0_pair[0],
    directors_ip1=directors_0_pair[1],
    t=0.5,
  )

  def _single_fiber_energy(
    fiber_offset: jnp.ndarray,
    fiber_shear_modulus: float,
    fiber_winding_direction: jnp.ndarray,
    fiber_reinforcement_modulus: Optional[float],
    remnant_mag_flux_k: Optional[jnp.ndarray],
    fiber_flory_param: Optional[float],
  ) -> jnp.ndarray:
    """Free energy density ψ [Pa] for one fiber at cross-section offset `fiber_offset`.

    Computes φ first (from the Flory-Rehner bisection or 1.0 for dry rods) and
    passes it to both `_compute_fiber_deformation_gradient` and `free_energy_neohookean`
    so that λ_t and ln(J) = ln(1/φ) are consistent with the same φ.

    Args:
      fiber_offset: Array of shape `(num_crossec_dim=2,)` [m]. Offsets (y₁, y₂)
        along d₁ and d₂ from the rod centerline.
      fiber_shear_modulus: Scalar float Gₖ [Pa]. Neo-Hookean shear modulus.
      fiber_winding_direction: Array of shape `(num_dim=3,)`. Unit reinforcement
        direction on the reference material axes. Ignored if `is_reinforced` is `False`.
      fiber_reinforcement_modulus: Scalar float ηₖ [Pa] or `None`. Reinforcement
        stiffness. `None` when `is_reinforced` is `False`.
      remnant_mag_flux_k: Array of shape `(num_dim=3,)` [T] or `None`. Per-fiber
        remnant flux in the reference global frame. `None` when `is_magnetic`
        is `False`.
      fiber_flory_param: Scalar float χₖ or `None`. Flory-Huggins parameter.
        `None` when `is_swelling` is `False`.

    Returns:
      Scalar float Πₖ [Pa]. Grand potential density for this fiber, per unit dry
        reference volume. For a swelling fiber this is the sum of the elastic
        Neo-Hookean energy, the Flory-Huggins energy of mixing, and the chemical
        work of solvent exchange with the bath,

             Π = ψ_el(λ, φ) + W_mix(φ) − (μ − μ⁰)·C(φ),

            where, W_mix(φ) = (RT/Ω)[((1−φ)/φ)·ln(1−φ) − χφ],
                   C(φ) = (1/φ − 1)/Ω.
    """
    phi = 1.0
    if is_swelling:
      lambda_axial, _, _ = _compute_fiber_kinematics(
        elem_kinematics=elem_kinematics,
        fiber_offset=fiber_offset,
        elem_ref_kinematics=elem_ref_kinematics,
        eps=eps,
      )
      phi = _compute_fiber_polymer_volume_fraction(
        total_fiber_axial_stretch=lambda_axial,
        flory_parameter=fiber_flory_param,
        shear_modulus=fiber_shear_modulus,
        chem_potential=chem_potential,
        solvent=solvent,
        bisection_settings=bisection_settings,
      )

    def_grad_material = _compute_fiber_deformation_gradient(
      elem_kinematics=elem_kinematics,
      fiber_offset=fiber_offset,
      polymer_vol_frac=phi,
      elem_ref_kinematics=elem_ref_kinematics,
      eps=eps,
    )

    energy = _material.free_energy_neohookean(
      def_grad=def_grad_material,
      shear_modulus=fiber_shear_modulus,
      polymer_vol_frac=phi,
      molecular_incompressibility=True,
      fiber_direction=fiber_winding_direction,
      reinforcement_modulus=fiber_reinforcement_modulus,
    )

    if is_swelling:
      r_theta = _utils.GAS_CONSTANT * solvent.bath_temperature
      solvent_frac = jnp.maximum(1.0 - phi, 1.0e-12)
      mixing_energy = (r_theta / solvent.molar_volume) * (
        (solvent_frac / phi) * jnp.log(solvent_frac) - fiber_flory_param * phi
      )
      solvent_concentration = (1.0 / phi - 1.0) / solvent.molar_volume
      chemical_energy = (
        -(chem_potential - solvent.ref_chem_potential) * solvent_concentration
      )

      energy = energy + mixing_energy + chemical_energy

    if is_magnetic:
      def_grad_global = _def_grad_material_to_global(
        def_grad_material=def_grad_material,
        directors_current=directors_mean_current,
        directors_ref=directors_mean_ref,
      )
      energy = energy + _material.free_energy_hard_magnetic(
        def_grad=def_grad_global,
        remnant_magnetic_flux_density=remnant_mag_flux_k,
        applied_magnetic_flux_density=fiber_applied_mag_flux,
      )

    return energy

  flory_in_ax = 0 if is_swelling else None
  rem_in_ax = 0 if is_magnetic else None
  reinf_in_ax = 0 if is_reinforced else None

  fiber_energy_densities = jax.vmap(
    _single_fiber_energy, in_axes=(0, 0, 0, reinf_in_ax, rem_in_ax, flory_in_ax)
  )(
    fiber_offsets,
    fiber_shear_moduli,
    fiber_winding_directions,
    fiber_reinforcement_moduli,
    fiber_remnant_mag_flux,
    fiber_flory_params,
  )  # (num_fibers,)

  return (
    jnp.einsum("f, f ->", fiber_energy_densities, fiber_areas) * elem_ref_arc_length
  )


def _compute_elem_internal_forces(
  elem_args: tuple,
  elem_in_axes: tuple,
) -> jnp.ndarray:
  """Compute the internal force vector of every element, fᵢₙₜ,ₑ = ∂Πₑ/∂uₑ.

  Differentiates `_compute_elem_energy` with respect to its first argument, the element
  dof vector, and vmaps over the elements.

  Args:
    elem_args: Tuple of the 16 positional arguments of `_compute_elem_energy`, batched
      over elements where `elem_in_axes` is 0, as built by `CosseratProblem`.
    elem_in_axes: Tuple of the 16 vmap axes matching `elem_args`, with 0 for batched
      arguments and `None` for shared or absent ones.

  Returns:
    Array of shape `(num_elems, dof_per_elem)` [N] or [N·m] of element internal forces.
  """
  elem_force_fn = jax.grad(_compute_elem_energy, argnums=0)
  return jax.vmap(elem_force_fn, in_axes=elem_in_axes)(*elem_args)


def _compute_elem_hessians(
  elem_args: tuple,
  elem_in_axes: tuple,
) -> jnp.ndarray:
  """Compute the energy Hessian of every element, Hₑ = ∂²Πₑ/∂uₑ².

  Forward-mode differentiation of the element internal force, fᵢₙₜ,ₑ = ∂Πₑ/∂uₑ, with
  respect to the element dof vector, vmapped over the elements. Serves both the Newton
  tangent, K_T,ₑ = −Hₑ, at the current state and the reference Hessian of the
  incremental regularization at uₑ = 0.

  Args:
    elem_args: Tuple of the 16 positional arguments of `_compute_elem_energy`, batched
      over elements where `elem_in_axes` is 0, as built by `CosseratProblem`.
    elem_in_axes: Tuple of the 16 vmap axes matching `elem_args`, with 0 for batched
      arguments and `None` for shared or absent ones.

  Returns:
    Array of shape `(num_elems, dof_per_elem, dof_per_elem)` of element Hessians.
  """
  elem_force_fn = jax.grad(_compute_elem_energy, argnums=0)
  elem_hessian_fn = jax.jacfwd(elem_force_fn, argnums=0)
  return jax.vmap(elem_hessian_fn, in_axes=elem_in_axes)(*elem_args)


class CosseratProblem:
  """Static equilibrium solver for a geometrically nonlinear 3D Cosserat rod.

  Discretizes the rod with 6 nodal DOFs per node [Δx, Δy, Δz, δθₓ, δθᵧ, δθᵤ],
  all increments from the reference configuration. The internal potential energy
  is assembled element-by-element and differentiated automatically to obtain the
  residual and tangent stiffness:

      R(u) = f_ext − ∂Π_int/∂u
      K_T(u) = ∂R/∂u = -∂²Π_int/∂u²

  Supported physics: elastic Neo-Hookean, Flory-Rehner gel swelling, hard-magnetic
  actuation, SDF-based wall contact, and self-contact.

  Multi-step (load-stepping) solves are supported. Each step warm-starts from
  a linear extrapolation of the two preceding converged states when its residual is
  finite, and from the previous converged state otherwise.


  Attributes:
    rod: `Rod` instance with the geometry, connectivity and other properties.
    bc: Processed `BoundaryConditions` containing assembled external forces, fixed dof
      indices, and Dirichlet increment values across time/load steps.
    nr_solver: `NewtonRaphson` instance handling nonlinear iteration solver.
    solvent: `FlorySolvent` dataclass containing the parameters for the Flory-Rehner
      swelling solvent. If `None`, the rod is non-swelling.
    bisection_settings: Optional dictionary of settings for the bisection method for
      the Flory-Rehner swelling solver. Pass `None` when not using the swelling model.
    obstacle: `Obstacle` providing the obstacle SDF and wall contact. When `None`, no
      no contact forces are applied.
    self_contact: `SelfContact` instance for modeling self-contact between rod elements.
        When `None`, no self-contact forces are applied.
    proximal_factor: Dimensionless γ ≥ 0 of the incremental regularization
      (γ/2) Σₑ Δuₑᵀ Hₑʳᵉᶠ Δuₑ, with Δuₑ the element increment from the previous
      converged load step and Hₑʳᵉᶠ the element's elastic Hessian at the stress-free
      reference. Active only in load-stepping solves. 0 disables it.
  """

  def __init__(
    self,
    rod: _mesher.Rod,
    bc: _bc.BoundaryConditions,
    nr_solver: _solver.NewtonRaphson,
    solvent: Optional[FlorySolvent] = None,
    bisection_settings: Optional[dict] = None,
    obstacle: Optional[_contact.Obstacle] = None,
    self_contact: Optional[_contact.SelfContact] = None,
    proximal_factor: float = 0.0,
  ) -> None:
    """Initializes the 3D Cosserat elastic rod FEA problem."""
    self.rod, self.bc = rod, bc
    self.nr_solver, self.bisection_settings = nr_solver, bisection_settings
    self.solvent, self.obstacle, self.self_contact = solvent, obstacle, self_contact
    self.proximal_factor = proximal_factor

  def _build_elem_args(
    self,
    u_elem: jnp.ndarray,
    shear_moduli: jnp.ndarray,
    reinforcement_moduli: Optional[jnp.ndarray] = None,
    remnant_mag_flux: Optional[jnp.ndarray] = None,
    flory_parameters: Optional[jnp.ndarray] = None,
    applied_mag_flux: Optional[jnp.ndarray] = None,
    chem_potential: Optional[jnp.ndarray] = None,
  ) -> Tuple[tuple, tuple]:
    """Build the element-batched arguments of `_compute_elem_energy` and vmap axes.

    Collects the element dof blocks, the reference node pairs and director pairs, the
    fiber data and the optional physics fields in the positional order expected by
    `_compute_elem_energy`. Optional fields passed as `None` get a `None` vmap axis, so
    the corresponding physics is switched off inside the element energy.

    Args:
      u_elem: Array of shape `(num_elems, dof_per_elem)`. Element dof blocks, zero at
        the stress-free reference.
      shear_moduli: Array of shape `(num_elems, num_fibers)` [Pa]. Neo-Hookean shear
        modulus per element per fiber.
      reinforcement_moduli: Array of shape `(num_elems, num_fibers)` [Pa] or `None`.
        Reinforcement stiffness per element per fiber.
      remnant_mag_flux: Array of shape `(num_elems, num_fibers, num_dim=3)` [T] or
        `None`. Remnant magnetic flux density in the reference global frame.
      flory_parameters: Array of shape `(num_elems, num_fibers)` or `None`.
        Flory-Huggins interaction parameters.
      applied_mag_flux: Array of shape `(num_dim=3,)` [T] or `None`. Applied magnetic
        flux density, shared by all elements.
      chem_potential: Scalar [J/mol] or `None`. Bath chemical potential, shared by all
        elements.

    Returns:
      A tuple `(elem_args, elem_in_axes)` of the 16 positional arguments of
        `_compute_elem_energy` and their vmap axes.
    """
    # (e)lement, (f)iber, d(o)f_per_elem, (n)ode, (d)im, (c)rosssec_dim_2
    ref_pos_pairs = jnp.stack(
      [self.rod.node_coords[:-1], self.rod.node_coords[1:]], axis=1
    )  # (e, n, d)
    directors_0_pairs = jnp.stack(
      [self.rod.directors_0[:-1], self.rod.directors_0[1:]], axis=1
    )  # (e, n, d, d)

    rem_in_ax = 0 if remnant_mag_flux is not None else None
    flory_in_ax = 0 if flory_parameters is not None else None
    reinf_in_ax = 0 if reinforcement_moduli is not None else None

    elem_args = (
      u_elem,  # (e, o) 1
      self.rod.elem_arc_lengths,  # (e,) 2
      ref_pos_pairs,  # (e, n, d) 3
      directors_0_pairs,  # (e, n, d, d) 4
      self.rod.ref_kinematics,  # (e, d) 5
      self.rod.fibers.offsets,  # (e, f, c) 6
      self.rod.fibers.areas,  # (e, f) 7
      shear_moduli,  # (e, f) 8
      self.rod.fibers.winding_directions,  # (e, f, d) 9
      reinforcement_moduli,  # (e, f) 10
      remnant_mag_flux,  # (e, f, d) 11
      flory_parameters,  # (e, f) 12
      applied_mag_flux,  # (d,) 13
      chem_potential,  # () 14
      self.solvent,  # () 15
      self.bisection_settings,  # () 16
    )
    elem_in_axes = (*[0] * 9, reinf_in_ax, rem_in_ax, flory_in_ax, *[None] * 4)
    return elem_args, elem_in_axes

  def compute_reference_elem_hessian(
    self,
    shear_moduli: jnp.ndarray,
    reinforcement_moduli: Optional[jnp.ndarray] = None,
  ) -> jnp.ndarray:
    """Computes every element's elastic energy Hessian at the stress-free reference.

    Evaluates Hₑʳᵉᶠ = ∂²Πₑ/∂uₑ² at uₑ = 0 with the current moduli and with magnetic,
    swelling and contact contributions absent. The reference state is stress-free, so
    Hₑʳᵉᶠ carries no geometric stiffness and is positive semidefinite, and it scales
    element by element with that element's own modulus, section and length. It
    depends on the moduli only, so a load-stepping solve evaluates it once.

    Args:
      shear_moduli: Array of shape `(num_elems, num_fibers)` [Pa]. Neo-Hookean shear
        modulus per element per fiber.
      reinforcement_moduli: Array of shape `(num_elems, num_fibers)` [Pa] or `None`.
        Reinforcement stiffness per element per fiber.

    Returns:
      Array of shape `(num_elems, dof_per_elem, dof_per_elem)` of element Hessians.
    """
    zero_elem_dofs = jnp.zeros(self.rod.elem_dof_mat.shape)  # (e, o)
    elem_args, elem_in_axes = self._build_elem_args(
      u_elem=zero_elem_dofs,
      shear_moduli=shear_moduli,
      reinforcement_moduli=reinforcement_moduli,
    )
    return _compute_elem_hessians(
      elem_args=elem_args,
      elem_in_axes=elem_in_axes,
    )

  def get_residual_and_tangent(
    self,
    u: jnp.ndarray,
    compute_jacobian: bool,
    shear_moduli: jnp.ndarray,
    reinforcement_moduli: Optional[jnp.ndarray] = None,
    remnant_mag_flux: Optional[jnp.ndarray] = None,
    flory_parameters: Optional[jnp.ndarray] = None,
    applied_mag_flux: Optional[jnp.ndarray] = None,
    chem_potential: Optional[jnp.ndarray] = None,
    bc_step_idx: int = 0,
    u_reference: Optional[jnp.ndarray] = None,
    proximal_hessian: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, Optional[jax_sprs.BCOO]]:
    """Compute the global residual and sparse tangent stiffness matrix.

    Implements the static equilibrium residual:

                    R(u) = f_ext − f_int(u)

    Dirichlet dofs are enforced by overwriting the corresponding entries of
    `u` with `dirichlet_values * load_factor` before assembly. The residual
    at constrained dofs is then zeroed out and the Jacobian rows and columns
    are replaced with the identity.

    Args:
      u: Array of shape `(num_dofs,)` = `(6·num_nodes,)`. Current global
        relative DOF vector. Entry layout per node i:
        [Δxᵢ, Δyᵢ, Δzᵢ, δθₓᵢ, δθᵧᵢ, δθᵤᵢ]. All entries are zero at the
        reference configuration.
      compute_jacobian: When `False`, only the residual is returned and `None`
        is returned for the Jacobian. Must be the second positional argument to
        interface with `NewtonRaphson.solve`.
      shear_moduli: Array of shape `(num_elems, num_fibers)` [Pa]. Neo-Hookean
        shear modulus per element per fiber.
      reinforcement_moduli: Optional array of shape `(num_elems, num_fibers)` [Pa].
        The transverse reinforcement of the fibers. For isotropic, set this to None or
        all zeros.
      remnant_mag_flux: Array of shape `(num_elems, num_fibers, num_dim=3)` [T]
        or `None`. Remnant magnetic flux density per element per fiber in the
        reference global Cartesian frame. Vmapped over elements when not `None`.
        Pass `None` for non-magnetic rods.
      flory_parameters: Array of shape `(num_elems, num_fibers)` [-]. Flory parameters
        for each element and fiber. Flory-Huggins interaction parameter χₖ ≥ 0 per
        fiber. Values near zero give maximal swelling (good solvent); values above 0.5
        suppress swelling (poor solvent). Pass `None` for dry elastic rods (φ = 1).
      applied_mag_flux: Array of shape `(num_dim=3,)` [T] or `None`. Applied
        magnetic flux density B_a in the current global Cartesian lab frame,
        shared across all elements. Pass `None` for non-magnetic rods.
      chem_potential: Scalar array of the chemical potential [J/mol] dictating the
        driving force for swelling. A large negative value indicates a dry environment
        while a value close to zero indicates a wet environment.
      bc_step_idx: Integer step index into `self.bc`. Defaults to 0 for
        single-step problems.
      u_reference: Array of shape `(num_dofs,)` or `None`. Previous converged state
        u_{k−1} with the current Dirichlet values imposed, anchoring the
        incremental regularization.
      proximal_hessian: Array of shape `(num_elems, dof_per_elem, dof_per_elem)` or
        `None`. γ Hₑʳᵉᶠ. Adds γ Hₑʳᵉᶠ (uₑ − uₑʳᵉᶠ) to the element internal force and
        −γ Hₑʳᵉᶠ to the element tangent. `None` disables it.

    Returns:
      A tuple containing:
        - Residual vector of shape `(num_dofs,)` [N] or [N·m].
        - Tangent stiffness in sparse `jax_sprs.BCOO` format of shape
          `(num_dofs, num_dofs)`. `None` when `compute_jacobian` is `False`.
    """
    dirichlet_mask_k = self.bc.dirichlet_mask[bc_step_idx, :]
    dirichlet_values_k = self.bc.dirichlet_values[bc_step_idx, :]
    u = jnp.where(dirichlet_mask_k, dirichlet_values_k, u)

    u_elem = u[self.rod.elem_dof_mat]  # (e, o)

    elem_args, elem_in_axes = self._build_elem_args(
      u_elem=u_elem,
      shear_moduli=shear_moduli,
      reinforcement_moduli=reinforcement_moduli,
      remnant_mag_flux=remnant_mag_flux,
      flory_parameters=flory_parameters,
      applied_mag_flux=applied_mag_flux,
      chem_potential=chem_potential,
    )
    elem_f_int = _compute_elem_internal_forces(
      elem_args=elem_args,
      elem_in_axes=elem_in_axes,
    )  # (e, o)

    if proximal_hessian is not None:
      reference_elem = u_reference[self.rod.elem_dof_mat]  # (e, o)
      elem_increment = u_elem - reference_elem
      elem_f_prox = jnp.einsum("eij, ej -> ei", proximal_hessian, elem_increment)
      elem_f_int = elem_f_int + elem_f_prox

    global_f_int = (
      jnp.zeros((self.rod.num_dofs,)).at[self.rod.elem_dof_mat].add(elem_f_int)
    )

    residual = self.bc.external_forces[bc_step_idx, :] - global_f_int

    if self.self_contact is not None:
      f_sc, k_sc = _contact.compute_self_contact_forces_and_tangent(
        u_elems=u_elem,
        rod=self.rod,
        self_contact=self.self_contact,
        compute_jacobian=compute_jacobian,
      )
      residual = residual - f_sc

    if self.obstacle is not None:
      obj_trans, obj_rot_mat = self.obstacle.pose_at_step(bc_step_idx)
      f_contact, k_contact = _contact.compute_wall_contact_forces_and_tangent(
        u_elems=u_elem,
        rod=self.rod,
        obstacle=self.obstacle,
        compute_jacobian=compute_jacobian,
        obj_trans=obj_trans,
        obj_rot_mat=obj_rot_mat,
      )
      residual = residual - f_contact

    residual = jnp.where(dirichlet_mask_k, 0.0, residual)

    if not compute_jacobian:
      return residual, None

    elem_jac = -_compute_elem_hessians(
      elem_args=elem_args,
      elem_in_axes=elem_in_axes,
    )  # (e, o, o)

    if proximal_hessian is not None:
      elem_jac = elem_jac - proximal_hessian

    node_idx = np.stack((self.rod.iK, self.rod.jK)).astype(np.int32).T
    assembled_jacobian = jax_sprs.BCOO(
      (elem_jac.flatten(order="C"), node_idx),
      shape=(self.rod.num_dofs, self.rod.num_dofs),
    )

    if self.obstacle is not None:
      assembled_jacobian = assembled_jacobian + k_contact

    if self.self_contact is not None:
      assembled_jacobian = assembled_jacobian + k_sc

    assembled_jacobian = _bc.apply_dirichlet_bc(
      jacobian=assembled_jacobian,
      dirichlet_mask=dirichlet_mask_k,
    )
    return residual, assembled_jacobian

  def solve(
    self,
    u0: jnp.ndarray,
    shear_moduli: jnp.ndarray,
    reinforcement_moduli: Optional[jnp.ndarray] = None,
    applied_mag_flux: Optional[jnp.ndarray] = None,
    remnant_mag_flux: Optional[jnp.ndarray] = None,
    flory_parameters: Optional[jnp.ndarray] = None,
    chem_potentials: Optional[jnp.ndarray] = None,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Solve static equilibrium problem via Newton-Raphson iteration.

    Supports single-step and multi-step (load-stepping) solves. For single-step
    problems (`self.bc.num_steps == 1`), calls `nr_solver.solve` once with the
    full load. For multi-step problems, ramps boundary conditions, applied
    magnetic flux, and chemical potential over `self.bc.num_steps` increments
    using `jax.lax.scan`. Each step warm-starts from a linear extrapolation of
    the two preceding converged states when that guess has a finite residual, and
    from the previous converged state otherwise. When `nr_solver.verbose`, every Newton
    solve prints its convergence line and a load-stepping solve ends with a summary
    over all steps (iteration range, final residual, unconverged steps).

    Args:
      u0: Array of shape `(num_dofs,)`. Initial solution guess. We solve for the
        nodal displacements from the reference configuration and the rotation of the
        nodes from the reference configuration.
      shear_moduli: Array of shape `(num_elems, num_fibers)` [Pa]. Neo-Hookean
        shear modulus per element per fiber. Constant across all load steps.
      reinforcement_moduli: Array of shape `(num_elems, num_fibers)` [Pa] or `None`.
        Transversely isotropic reinforcement stiffness ηₖ ≥ 0 per element per fiber,
        acting along the fiber winding directions stored on the rod. Constant across
        all load steps. Pass `None` for a purely isotropic rod, which reproduces the
        unreinforced response exactly.
      applied_mag_flux: Array of shape `(num_dim=3,)` [T] for a single load step or
        `(num_load_steps, num_dim=3)` [T] for a load-stepping solve. The applied
        magnetic flux density B_a in the current global Cartesian lab frame. Pass
        `None` for non-magnetic rods.
      remnant_mag_flux: Array of shape `(num_elems, num_fibers, num_dim=3)` [T] or
        `None`. Remnant magnetic flux density per element per fiber in the
        reference global Cartesian frame. Pass `None` for non-magnetic rods.
      flory_parameters: Array of shape `(num_elems, num_fibers)` of floats χₖ ≥ 0.
        Flory-Huggins interaction parameter per element per fiber. A value close
        to zero corresponds to a good solvent with high swelling, while larger values
        (χ > 0.5) correspond to poorer solvents with less swelling. If `None`,
        solves without swelling by setting φ=1 (dry state).
      chem_potentials: Array of shape `(num_steps,)` [J/mol] or `None`. Bath
        chemical potential μ at each step. Ramped from a large negative dry value to the
        target wet value to drive incremental swelling. Pass `None` for dry elastic
        rods. When `None`, solves without swelling.

    Returns:
      u_star: Array of shape `(num_dofs,)`. Converged dof vector at full load.
      u_history: Array of shape `(num_load_steps, num_dofs)`. Converged dofs
        at each load step. `None` when self.bc.num_steps == 1.
    """
    if self.bc.num_steps == 1:
      chem_potential = chem_potentials[0] if chem_potentials is not None else None
      u_star, _ = self.nr_solver.solve(
        self.get_residual_and_tangent,
        u0,
        shear_moduli,
        reinforcement_moduli,
        remnant_mag_flux,
        flory_parameters,
        applied_mag_flux,
        chem_potential,
        0,
      )
      return u_star, None

    num_steps = self.bc.num_steps
    u_hist = jnp.zeros((num_steps, self.rod.num_dofs))

    proximal_hessian = None
    if self.proximal_factor > 0.0:
      reference_hessian = self.compute_reference_elem_hessian(
        shear_moduli=shear_moduli,
        reinforcement_moduli=reinforcement_moduli,
      )
      proximal_hessian = self.proximal_factor * reference_hessian

    def _load_step(
      carry: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
      step_idx: int,
    ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray], _solver.NRStats]:
      """Single load step: pick a finite starting guess, solve, record."""
      u_prev, u_prev_prev, u_hist = carry
      dirichlet_mask_k = self.bc.dirichlet_mask[step_idx, :]
      dirichlet_values_k = self.bc.dirichlet_values[step_idx, :]
      u_reference = jnp.where(dirichlet_mask_k, dirichlet_values_k, u_prev)

      app_mag = applied_mag_flux[step_idx, :] if applied_mag_flux is not None else None
      chem_pot = chem_potentials[step_idx] if chem_potentials is not None else None

      u_extrapolated = self.rod.wrap_rotational_dofs(2.0 * u_prev - u_prev_prev)
      extrapolated_guess = jax.lax.stop_gradient(
        jnp.where(dirichlet_mask_k, dirichlet_values_k, u_extrapolated)
      )
      anchor_guess = jax.lax.stop_gradient(u_reference)
      extrapolated_residual, _ = self.get_residual_and_tangent(
        u=extrapolated_guess,
        compute_jacobian=False,
        shear_moduli=shear_moduli,
        reinforcement_moduli=reinforcement_moduli,
        remnant_mag_flux=remnant_mag_flux,
        flory_parameters=flory_parameters,
        applied_mag_flux=app_mag,
        chem_potential=chem_pot,
        bc_step_idx=step_idx,
        u_reference=anchor_guess,
        proximal_hessian=proximal_hessian,
      )
      extrapolation_is_finite = jnp.all(jnp.isfinite(extrapolated_residual))
      use_extrapolation = (step_idx > 0) & extrapolation_is_finite
      u_guess = jnp.where(use_extrapolation, extrapolated_guess, anchor_guess)

      u_next, nr_stats = self.nr_solver.solve(
        self.get_residual_and_tangent,
        u_guess,
        shear_moduli,
        reinforcement_moduli,
        remnant_mag_flux,
        flory_parameters,
        app_mag,
        chem_pot,
        step_idx,
        u_reference,
        proximal_hessian,
      )
      u_next = self.rod.wrap_rotational_dofs(u_next)

      u_hist = u_hist.at[step_idx, :].set(u_next)
      return (u_next, u_prev, u_hist), nr_stats

    (u_star, _, u_history), nr_stats = jax.lax.scan(
      _load_step,
      (u0, u0, u_hist),
      jnp.arange(num_steps),
    )

    self.nr_solver.report(nr_stats)  # summary over all load steps
    return u_star, u_history
