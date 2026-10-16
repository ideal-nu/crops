"""Materials and constitutive models for cosserat rod analysis."""

import dataclasses
from typing import Optional

import jax.numpy as jnp

import crops.src.utils as _utils


@dataclasses.dataclass
class StructuralMaterial:
  """Structural material constants.

  Attributes:
    youngs_modulus: The young's modulus of the material [Pa].
    poissons_ratio: The poisson's ratio of the material [-].
    mass_density: Mass density of material in [kg/m^3].
    yield_strength: Yield strength of the material [Pa].
  """

  youngs_modulus: Optional[float] = None
  poissons_ratio: Optional[float] = None
  mass_density: Optional[float] = None
  yield_strength: Optional[float] = None

  @property
  def shear_modulus(self) -> float:
    return self.youngs_modulus / (2.0 * (1.0 + self.poissons_ratio))

  @property
  def bulk_modulus(self) -> float:
    if self.poissons_ratio >= 0.5:  # Incompressible material
      return float("inf")
    return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poissons_ratio))

  @property
  def lame_parameters(self) -> tuple[float, float]:
    """Get the Lame parameters for the material.

    Returns: The Lame parameters as a tuple (lambda, mu) for the material.
    """
    mu = self.shear_modulus
    if self.bulk_modulus == float("inf"):  # Incompressible material
      return float("inf"), mu

    lam = (
      self.youngs_modulus
      * self.poissons_ratio
      / ((1.0 + self.poissons_ratio) * (1.0 - 2.0 * self.poissons_ratio))
    )

    return lam, mu


def free_energy_neohookean(
  def_grad: jnp.ndarray,
  shear_modulus: float,
  polymer_vol_frac: float = 1.0,
  molecular_incompressibility: bool = True,
  fiber_direction: Optional[jnp.ndarray] = None,
  reinforcement_modulus: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Helmholtz strain energy density for neohookean (with optional fiber anisotropy).

  The Helmholtz free energy density (ψ) for isotropic is given by:
          ψ = G/2 · (I₁ − 3 − 2 ln J)

  where I₁ = tr(C) is the first invariant of the right Cauchy-Green tensor `C = F^T F`,
  and J = det(F) is the determinant of the deformation gradient.

  This covers incompressible and compressible neohookean materials as well as with
  or without swelling.

  φ = 1, incompress = True ->  ψ = G/2 · (I₁ − 3) ; Incompressible neohookean J = 1
  φ < 1, incompress = True ->  ψ = G/2 · (I₁ − 3 + 2 ln φ) ; swelling gel, J = 1/φ
  φ <= 1, incompress = False ->  ψ = G/2 · (I₁ − 3 − 2 ln J) ; compressible J = det(F)

  When both `fiber_direction` and `reinforcement_modulus` are supplied, a transversely
  isotropic reinforcement term is added,

              ψ_aniso = 0.5 * eta_f * (I_4 - 1)^2,
              I_4 = a . C a = ||F a||^2,

  where `a` is the unit reinforcement direction in the reference configuration and `I_4`
  is the squared stretch of that direction.

  Args:
    def_grad: The deformation gradient tensor of shape (num_dim=3, num_dim=3).
    shear_modulus: The shear modulus mu (= G).
    polymer_vol_frac: Scalar float φ ∈ (0, 1]. Polymer volume fraction. φ = 1 is the
      fully dry state; φ → 0 is the fully swollen limit. Ignored when
      `molecular_incompressibility` is `False`.
    molecular_incompressibility: Static bool. When `True`, the volume ratio is
      prescribed as J = 1/φ by molecular incompressibility of the polymer-solvent
      mixture. When `False`, it is computed as J = det(F). Defaults to `True`.
    fiber_direction: Optional array of shape `(num_dim=3,)`. Unit reinforcement
      direction in the reference configuration, resolved on the same axes as `def_grad`.
      The caller is responsible for normalisation. Pass `None` for isotropic material.
    reinforcement_modulus: Optional scalar float eta_f [Pa], with eta_f >= 0. Stiffness
      of the reinforcing phase. A value of zero recovers the isotropic response exactly
       is the correct setting for a regression test. Pass `None` for isotropic material.

  Returns:
    Scalar elastic free energy density ψ [Pa].
  """
  right_cauchy_green = jnp.einsum("ij, ik -> jk", def_grad, def_grad)  # C = F^T F
  first_invariant = jnp.trace(right_cauchy_green)

  if molecular_incompressibility:  # J = 1/φ  ⇒  ln J = −ln φ
    log_volume_ratio = -_utils.safe_log(polymer_vol_frac)
  else:
    log_volume_ratio = _utils.safe_log(jnp.linalg.det(def_grad))

  energy = 0.5 * shear_modulus * (first_invariant - 3.0 - 2.0 * log_volume_ratio)

  if fiber_direction is None or reinforcement_modulus is None:
    return energy

  fourth_invariant = jnp.einsum(
    "i, ij, j ->", fiber_direction, right_cauchy_green, fiber_direction
  )
  return energy + 0.5 * reinforcement_modulus * (fourth_invariant - 1.0) ** 2


def free_energy_hard_magnetic(
  def_grad: jnp.ndarray,
  remnant_magnetic_flux_density: jnp.ndarray,
  applied_magnetic_flux_density: jnp.ndarray,
  permeability: jnp.ndarray = _utils.PERMEABILITY_VACUUM,
) -> jnp.ndarray:
  """Compute the Helmholtz free energy density induced by remnant magnetic fields.

    The Helmholtz free energy density (psi) induced by magnetic fields is given by:

              psi = - (1/permeability) *  B_a . (F B_r)

    where F is the deformation gradient, B_a is the applied magnetic flux density,
    B_r is remnant magnetic flux density.

    For more details, see eq 2 in:
      Zhao, Zhi, and Xiaojia Shelly Zhang. "Topology optimization of hard-magnetic soft
      materials." Journal of the Mechanics and Physics of Solids 158 (2022): 104628.

  NOTE: All three tensor and vector arguments must be resolved on the global Cartesian
  axes. The deformation gradient for cosserat rods is assembled in the co-rotating
  material frame {d₁, d₂, d₃}; convert it with `_def_grad_material_to_global` before
  calling. Likewise if the remanent flux density specified along the material directors
  (e.g. magnetized along d₃) must be rotated to global components.

  Args:
    def_grad: Array of shape (num_dim=3, num_dim=3) containing deformation gradient
      tensor.
    remnant_magnetic_flux_density: Array of shape (num_dim=3,) containing the remnant
      magnetic flux density vector (Tesla).
    applied_magnetic_flux_density: Array of shape (num_dim=3,) containing the applied
      magnetic flux density vector (Tesla).
    permeability: The permeability of the material. Default is the permeability of
      vacuum (4.0 * pi * 1.0e-7 Henries/meter).

  Returns: A scalar array of the Helmholtz free energy density induced by the magnetic
    fields.
  """
  return -(1.0 / permeability) * jnp.einsum(
    "d, di, i ->",
    applied_magnetic_flux_density,
    def_grad,
    remnant_magnetic_flux_density,
  )
