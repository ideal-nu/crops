"""Cosserat rod mesher."""

import dataclasses
import enum
import functools as ft
from typing import Callable, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax_dataclasses as jxd
import numpy as np

import crops.src.utils as _utils

_FLOAT_TO_ARRAY_FN = Callable[[float], jnp.ndarray]


@jxd.pytree_dataclass
class BoundingBox:
  x: _utils.Extent
  y: Optional[_utils.Extent] = None
  z: Optional[_utils.Extent] = None

  @property
  def num_dim(self) -> int:
    """Returns the spatial dimension count of the bounding box."""
    return 1 + int(self.y is not None) + int(self.z is not None)

  @property
  def min(self) -> jnp.ndarray:
    """Returns the minimum coordinate vector of the bounding box."""
    return jnp.array(
      [self.x.min, self.y.min if self.y else 0.0, self.z.min if self.z else 0.0]
    )[: self.num_dim]

  @property
  def max(self) -> jnp.ndarray:
    """Returns the maximum coordinate vector of the bounding box."""
    return jnp.array(
      [self.x.max, self.y.max if self.y else 0.0, self.z.max if self.z else 0.0]
    )[: self.num_dim]

  @property
  def lx(self) -> jnp.ndarray:
    """Returns the extent of the bounding box in the global x direction."""
    return jnp.abs(self.x.range)

  @property
  def ly(self) -> jnp.ndarray:
    """Returns the extent of the bounding box in the global y direction."""
    return jnp.abs(self.y.range) if self.y else 0.0

  @property
  def lz(self) -> jnp.ndarray:
    """Returns the extent of the bounding box in the global z direction."""
    return jnp.abs(self.z.range) if self.z else 0.0

  @property
  def diag_length(self) -> jnp.ndarray:
    """Returns the Euclidean length of the bounding box diagonal."""
    return jnp.sqrt(self.lx**2 + self.ly**2 + self.lz**2)


@jxd.pytree_dataclass
class Fibers:
  """Container for cross-sectional fiber geometries along a rod.

  Assumptions:
    1. Each rod is constructed with an internal assembly of sub-fibers. Fibers run
        along the longitudinal extent of the rod element grid. We assume the rod (and
        fibers) run along the d₃ tangent direction, with the cross-section spanned by
        the d₁ and d₂ material directors.
    2. Each fiber is discretised at the element level, so fiber properties can vary
        along the rod length.
    3. Deformations follow the kinematic assumption that cross-sections remain
        planar but not necessarily normal to the centerline (allowing shear
        and twist).

  Attributes:
    offsets: Array of shape (num_elems, num_fibers, num_crossec_dim=2). Local cross-sec
      offsets of each fiber centroid from the rod centerline, measured in the
      material frame. `offsets[e, f, 0]` is the offset along d₁, `offsets[e, f, 1]` is
      the offset along d₂. These are material-frame scalars; i.e., they are defined in
      local material frame and independent of the global frame orientation of the rod.
    areas: Array of shape (num_elems, num_fibers) of the cross-sectional areas of each
      fiber along the length (discretized by the elements).
    winding_angles: Array of shape (num_elems, num_fibers) [rad]. Winding angle alpha
      of each fiber, measured in the reference material frame from the rod tangent d₃
      toward the counter-clockwise circumferential direction e_theta = d₃ × e_r. Zero
      denotes a fiber running parallel to the rod axis, which is the default and which
      recovers a purely isotropic response. A positive value traces a right-handed
      helix.
    winding_directions: Array of shape (num_elems, num_fibers, num_dim=3). Unit
      reinforcement direction of each fiber on the reference material axes
      (d₁, d₂, d₃), precomputed once from `offsets` and `winding_angles` by
      `compute_fiber_winding_directions`. Stored rather than recomputed because it is
      pure reference geometry, exactly like `positions`.
    positions: Array of shape (num_elems, num_fibers, num_dim=3) of the absolute 3D
      spatial positions of each sub-fiber segment evaluated at the element centers. This
      corresponds to the global frame coordinates of the fiber centroids in the initial
      undeformed configuration.
    fiber_mass_density: mass density array of shape (num_elems, num_fibers) with the
      mass density values [kg/m3].
    centroid_moi: Array of shape (num_elems, num_fibers, num_crossec_dim=2) storing the
      centroidal area MOI for every fiber. Index 0 stores the MOI about the local d₁
      axis (resisting bending out of the d₁-d₃ plane), and Index 1 stores the MOI about
      the local d₂ axis (resisting bending out of the d₂-d₃ plane).
  """

  offsets: jnp.ndarray
  areas: jnp.ndarray
  winding_angles: jnp.ndarray
  winding_directions: jnp.ndarray
  positions: jnp.ndarray
  fiber_mass_density: jnp.ndarray
  centroid_moi: jnp.ndarray

  @property
  def num_elems(self) -> int:
    """Returns the total number of discrete elements along the rod axis."""
    return self.offsets.shape[0]

  @property
  def num_fibers(self) -> int:
    """Returns the total number of internal sub-fibers in the cross-section."""
    return self.offsets.shape[1]

  def nodal_offsets(
    self,
    elem_arc_lengths: jnp.ndarray,
  ) -> jnp.ndarray:
    """Interpolate element-level fiber offsets to nodes via arc-length weighting.

    For an interior node n shared by element n-1 (arc length Δŝ_{n-1}) and
    element n (arc length Δŝ_n), we have:

        offset[n] = (Δŝ_n · offsets[n-1]  +  Δŝ_{n-1} · offsets[n])
                    ─────────────────────────────────────────────────
                                Δŝ_{n-1} + Δŝ_n

    Boundary nodes take the value of their single adjacent element directly.

    Args:
      elem_arc_lengths: Array of shape `(num_elems,)` [m]. Reference arc-length
        of each element. This corresponds to the arc coordinate "s" of the rod
        centerline in the reference configuration.

    Returns:
      Array of shape `(num_nodes, num_fibers, num_crossec_dim=2)` [m].
    """
    denom = elem_arc_lengths[:-1] + elem_arc_lengths[1:]
    weight_left = elem_arc_lengths[1:] / denom
    weight_right = elem_arc_lengths[:-1] / denom

    interior = jnp.einsum("n, nfi -> nfi", weight_left, self.offsets[:-1]) + jnp.einsum(
      "n, nfi -> nfi", weight_right, self.offsets[1:]
    )  # (num_elems - 1, num_fibers, 2)

    return jnp.concatenate(
      [self.offsets[:1], interior, self.offsets[-1:]], axis=0
    )  # (num_nodes, num_fibers, 2)


def compute_fiber_positions(
  node_coords_ref: jnp.ndarray,
  directors_ref: jnp.ndarray,
  fiber_offsets: jnp.ndarray,
  node_displacements: Optional[jnp.ndarray] = None,
  incremental_rotations: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
  """Compute global 3D fiber centroid positions at every node.

  Returns the position of each fiber centroid in the global Cartesian frame:

      r_fiber[n, f] = (r⁰[n] + Δr[n]) +  y₁[n,f] · d₁[n]  +  y₂[n,f] · d₂[n]

  All inputs share the leading node axis. When both optional arguments are `None`,
  returns the reference (undeformed) fiber positions.

  Frame convention for `fiber_offsets`:
    `fiber_offsets[n, f, 0]` = y₁ and `fiber_offsets[n, f, 1]` = y₂ are Lagrangian
    material-frame offsets, i.e., offsets along d₁ and d₂ respectively. They are NOT
    global Cartesian offsets, and they are invariant under deformation including twist.

  Args:
    node_coords_ref: Array of shape `(num_nodes, num_dim=3)` [m] of the reference
      centerline positions of the nodes defining the fiber in the global frame.
    directors_ref: Array of shape `(num_nodes, num_dirs=3, num_dim=3)` of the reference
      director frames (undeformed config). `directors_ref[n, k, :]` is the k-th material
      director dₖ⁰ at node n. Rows are orthonormal.
    fiber_offsets: Array of shape `(num_nodes, num_fibers, num_crossec_dim=2)`
      [m]. Node-level reference material-frame offsets.
    node_displacements: Optional array of shape `(num_nodes, num_dim=3)` [m].
      Translational displacement increments Δr at each node from the reference
      configuration. When `None`, reference centerline positions are used.
    incremental_rotations: Optional array of shape `(num_nodes, num_dim=3)`
      [rad]. Incremental rotation vectors δθ at each node in the global
      Cartesian frame. When `None`, reference directors are used unchanged.

  Returns:
    Array of shape `(num_nodes, num_fibers, num_dim=3)` [m]. Global 3D
    position of each fiber centroid at each node. Reduces to the reference
    fiber positions when both optional arguments are `None`.
  """
  # (n)odes, (d)im, (f)ibers, (c)ross-section dim, di(r)ectors
  node_coords_current = (
    node_coords_ref
    if node_displacements is None
    else node_coords_ref + node_displacements
  )

  directors_current = (
    directors_ref
    if incremental_rotations is None
    else jax.vmap(_utils.rotate_directors)(
      rotation=incremental_rotations,
      ref_directors=directors_ref,
    )
  )  # (n, r, d)

  d1 = directors_current[:, _utils.Director.D1, :]  # (n, d)
  d2 = directors_current[:, _utils.Director.D2, :]  # (n, d)

  return (
    node_coords_current[:, None, :]
    + jnp.einsum("nf, nd -> nfd", fiber_offsets[..., 0], d1)
    + jnp.einsum("nf, nd -> nfd", fiber_offsets[..., 1], d2)
  )


def compute_fiber_winding_directions(
  fiber_offsets: jnp.ndarray,
  winding_angles: jnp.ndarray,
  eps: float = 1e-12,
) -> jnp.ndarray:
  """Build the unit reinforcement direction of every helically wound fiber.

  A fiber wound at angle `α` about the rod axis follows a helix on the cylinder of
  radius `r` passing through its centroid. Its reference material
  direction therefore lies in the plane spanned by the rod tangent and the
  local circumferential direction at that centroid,

              a = cos(α) * d_3 + sin(α) * e_theta.

  The construction proceeds in three steps.

    1. Resolve the radial distance of the fiber centroid from the centerline,
       `r = sqrt(y_1^2 + y_2^2)`, where `(y_1, y_2)` are the material-frame
       offsets along `d_1` and `d_2`. The outward radial unit vector is then
       `e_r = (y_1, y_2, 0) / r`.
    2. Rotate `e_r` by a quarter turn within the cross-sectional plane to
       obtain the circumferential unit vector, `e_theta = d_3 x e_r =
       (-y_2, y_1, 0) / r`. This is the direction of travel of a point
       circling the rod axis counter-clockwise at fixed radius, and it
       satisfies `e_r . e_theta = 0` identically.
    3. Combine the axial and circumferential components with the cosine and
       sine of the winding angle. The result is automatically of unit length,
       since `cos^2(alpha) + sin^2(alpha) * ||e_theta||^2 = 1`.

  The sign convention matters, because handedness is the entire purpose of
  this quantity. A positive `alpha` winds toward positive `e_theta` while
  advancing along positive `d_3`, which traces a right-handed helix. Reversing
  the sign of `alpha` reverses the handedness and reverses the sign of the
  extension-torsion coupling that the reinforcement subsequently produces.

  Observe that the circumferential direction is undefined for a fiber lying on
  the rod axis, where `r` vanishes and there is no circumference to wind
  around. Such a fiber cannot be helical at any angle, so its direction is
  forced to the pure tangent `d_3`, which is the only orientation that remains
  well posed. Both the radial division and the axial component are guarded, so
  that the returned vector stays of unit length rather than collapsing to
  `(0, 0, cos(alpha))`. The guard uses the two-sided `jnp.where` idiom because
  reverse-mode differentiation propagates a non-finite cotangent through the
  untaken branch of a division by zero even when that branch is masked out.

  Args:
    fiber_offsets: Array of shape `(num_elems, num_fibers, num_crossec_dim=2)`
      [m]. Material-frame offsets `(y_1, y_2)` of each fiber centroid along
      `d_1` and `d_2`, as stored on `Fibers.offsets`.
    winding_angles: Array of shape `(num_elems, num_fibers)` [rad]. Winding
      angle alpha of each fiber measured from the rod tangent `d_3`, positive
      toward the counter-clockwise circumferential direction. A value of zero
      gives a purely tangential reinforcement and carries no coupling. Values
      approaching plus or minus pi/2 give a purely circumferential
      reinforcement, which likewise carries no coupling, so useful magnitudes
      lie strictly between those limits and peak at pi/4.
    eps: Scalar float. Floor on the radial distance below which a fiber is
      treated as lying on the rod axis.

  Returns:
    Array of shape `(num_elems, num_fibers, num_dim=3)`. Unit reinforcement
    direction of every fiber, resolved on the reference material axes
    `(d_1, d_2, d_3)`.
  """
  d1_offsets = fiber_offsets[..., _utils.Director.D1]  # (e, f)
  d2_offsets = fiber_offsets[..., _utils.Director.D2]  # (e, f)

  radii = _utils.safe_sqrt(d1_offsets**2 + d2_offsets**2)  # (e, f)
  is_on_axis = radii < eps
  safe_radii = jnp.where(is_on_axis, 1.0, radii)

  circum_scale = jnp.where(is_on_axis, 0.0, jnp.sin(winding_angles) / safe_radii)
  axial_scale = jnp.where(is_on_axis, 1.0, jnp.cos(winding_angles))

  return jnp.stack(
    [-d2_offsets * circum_scale, d1_offsets * circum_scale, axial_scale], axis=-1
  )


class StrainComponent(enum.IntEnum):
  """Integer indices mapping position-gradient strain components.

    The strain vector `strain_vec = [η₁, η₂, ν]` represents the projection of the
    centerline derivative ∂r/∂s onto the local material frame:

      η₁ = (∂r/∂s) · d₁   (Transverse shear along d₁)
      η₂ = (∂r/∂s) · d₂   (Transverse shear along d₂)
      ν  = (∂r/∂s) · d₃   (Axial stretch along axial director d₃)

  Assumptions:
    1. Strain quantities are packed as a 3-element vector [η₁, η₂, ν].

  Attributes:
    SHEAR_D1: Index 0. Transverse shear along the local d₁ director (η₁). Evaluates
      to 0.0 at the stress-free reference state.
    SHEAR_D2: Index 1. Transverse shear along the local d₂ director (η₂). Evaluates
      to 0.0 at the stress-free reference state.
    AXIAL_STRETCH: Index 2. Axial stretch along the axial director d₃ (ν). Evaluates
      to 1.0 at the stress-free reference state.

  """

  SHEAR_D1 = 0
  SHEAR_D2 = 1
  AXIAL_STRETCH = 2


class CurvatureComponent(enum.IntEnum):
  """Integer indices mapping discrete Darboux curvature components.

    The curvature vector `[κ₁, κ₂, κ₃]` represents the local Darboux vector generator
    κ̄, tracking the spatial rate of rotation of the material frame along the arc length:
      (∂ₛd̄ⱼ = κ̄ × d̄ⱼ)
      => κ₁ =  -(∂d₃/∂s) · d₂   (Bending curvature in the d₂-d₃ plane)
      => κ₂ =  -(∂d₁/∂s) · d₃   (Bending curvature in the d₁-d₃ plane)
      => κ₃ =  -(∂d₂/∂s) · d₁   (Torsional twist rate about the axial director d₃)

    NOTE: κᵢ is the rotation rate about dᵢ, so it bends the centerline in the plane
      perpendicular to dᵢ.

  Assumptions:
    1. Curvature quantities are packed as a 3-element vector [κ₁, κ₂, κ₃].

  Attributes:
    BENDING_D1: Index 0. Bending curvature about d₁ (κ₁); bends the centerline in
      the d₂-d₃ plane. Evaluates to 0.0 at the stress-free reference state.
    BENDING_D2: Index 1. Bending curvature about d₂ (κ₂); bends the centerline in
      the d₁-d₃ plane. Evaluates to 0.0 at the stress-free reference state.
    TWIST: Index 2. Torsional twist rate about the axial director d₃ (κ₃).
      Evaluates to 0.0 at the stress-free reference state.
  """

  BENDING_D1 = 0
  BENDING_D2 = 1
  TWIST = 2


@jxd.pytree_dataclass
class CosseratKinematics:
  """Container for element-level discrete Cosserat rod kinematic measures.

    The kinematic state of elements tracks two vector fields :

      ε = (∂r/∂s) − d₃        (shear-and-stretch vector; material components
                               [η₁, η₂, ν − 1], zero at rest)
      ∂ₛdⱼ = κ × dⱼ           (curvature vector as Darboux generator)

    Using discrete finite differences over the reference arc length Δŝ:
      strain_vec = [η₁, η₂, ν]      (components of ∂r/∂s, not of ε; [0, 0, 1] at rest)
      curvature_vec = [κ₁, κ₂, κ₃]

  Assumptions:
    1. The reference configuration is a straight, untwisted rod, for which the
       stress-free values are exactly strain_vec[e] = [0, 0, 1] and
       curvature_vec[e] = [0, 0, 0]. For a naturally curved or pre-twisted
       reference the elastic measure is (curvature − curvature_ref), not
       curvature itself.

  Attributes:
    strain: Array of shape (num_elems, 3) storing [η₁, η₂, ν] per element, in the
      element material frame. Equals [0, 0, 1] in the stress-free reference state.
    curvature: Array of shape (num_elems, 3) storing [κ₁, κ₂, κ₃] per element [1/m],
      in the element material frame. Equals [0, 0, 0] in the stress-free ref. state.
  """

  strain: jnp.ndarray
  curvature: jnp.ndarray


def compute_discrete_cosserat_kinematics(
  node_coords: jnp.ndarray,
  directors: jnp.ndarray,
  arc_coords: jnp.ndarray,
) -> CosseratKinematics:
  """Compute element-level discrete Cosserat kinematic measures from nodal data.

    The state at arc-length s is parameterized by a centerline r(s) and an
    orthonormal material frame {d₁(s), d₂(s), d₃(s)}. The six kinematic strain
    scalars decompose into position-gradient and Darboux curvature vectors:
      strain_vec = [η₁, η₂, ν]
      curvature_vec = [κ₁, κ₂, κ₃]

    In the discrete setting, derivatives are evaluated via finite differences
    over the reference arc length interval Δŝₑ = arc_coords[e+1] − arc_coords[e].
    Let d̂_mean be the SO(3) geodesic mean frame between element boundary nodes.

    1. Position-gradient vector (shear and stretch):
         strain[e, k] = (Δrₑ · d̂ₖ_mean) / Δŝₑ
    2. Darboux curvature vector (bending and twist):
          r_rel[e] = directors[e] @ directors[e+1].T is the rotation carrying frame e
          to frame e+1, expressed in the body (material) frame. Integrating
          ∂ₛdⱼ = κ × dⱼ across one element gives r_rel[e] = exp(Δŝₑ [κ]_×), so the
          matrix logarithm is Δŝₑ·κ resolved on {d₁, d₂, d₃}:

              [κ₁, κ₂, κ₃] = log(r_rel[e]) / Δŝₑ

    Both measures are second-order accurate at the element midpoint.

  Args:
    node_coords: Discrete nodal centerline positions of shape (num_nodes, num_dim=3)
      [m], in the current configuration.
    directors: Orthonormal material frame of shape (num_nodes, num_vec=3, num_dim=3)
      in the current configuration. Row 0 is d₁, row 1 is d₂, row 2 is d₃.
    arc_coords: Cumulative arc coordinates of shape (num_nodes,) [m] in the reference
      configuration. Only successive differences are used.

  Returns:
    A `CosseratKinematics` with `strain` and `curvature` each of shape (num_elems, 3).
    `curvature` is the Darboux vector: κᵢ is the rotation rate about dᵢ.
  """

  chords = node_coords[1:] - node_coords[:-1]
  elem_arc_lengths = arc_coords[1:] - arc_coords[:-1]

  mean_directors = jax.vmap(_utils.interpolate_directors_geodesic_so3)(
    directors[:-1], directors[1:]
  )

  strain_vec = (
    jnp.einsum("ei, eki -> ek", chords, mean_directors) / elem_arc_lengths[:, None]
  )

  r_rel = jnp.einsum("eia, eja -> eij", directors[:-1], directors[1:])
  theta_vecs = jax.vmap(_utils.rotation_matrix_to_vector)(r_rel)

  curvature_vec = theta_vecs / elem_arc_lengths[:, None]

  return CosseratKinematics(
    strain=strain_vec,
    curvature=curvature_vec,
  )


@dataclasses.dataclass
class Rod:
  """Data container storing a Cosserat rod.

  The rod is stored as a collection of discrete element segments. Each rod can
  be thought of as a 1D curve in 3D space with internal fibers defining the
  cross-sectional geometry. The degrees of freedom for a rod are packed into a
  sequential layout:

    X_rod = [ Δx₁, Δy₁, (Δz₁), δθ₁, ..., Δx_{N+1}, Δy_{N+1}, (Δz_{N+1}), δθ_{N+1} ]ᵀ

  Both translational and rotational dofs are attributed to nodes. Each node block has
  all translational components followed by all rotational components.
  For 3D: [x, y, z, θx, θy, θz] per node (6 dofs). For 2D: [x, y, θz] per
  node (3 dofs). These are relative position and orientation values. The translational
  components represent the incremental displacements of the centerline nodes from the
  reference configuration. The rotational components represent the incremental rotation
  of the material frame directors from the reference configuration. The reference
  configuration is defined by the `node_coords` and `directors_0` fields, which
  represent the stress-free state of the rod. The reference configuration can be curved
  and pre-twisted, so the reference strains are not necessarily zero.

  Assumptions:
    1. The structural assembly consists of `e` elements bounded by `e + 1`
        nodes.
    2. The rest `directors_0` define the natural curvature and twist profiles
        from which elastic deformations generate strain energy penalties. The
        rod is stress-free when its directors match `directors_0`.

  Attributes:
    node_coords: Array of shape (num_nodes, num_dim=3) of global 3D Cartesian
      coordinates of the centerline nodes.
    arc_coords: Array of shape (num_nodes,) of cumulative Euclidean chord-length
      coordinates. arc_coords[0] = 0 and arc_coords[e+1] − arc_coords[e] =
      ‖node_coords[e+1] − node_coords[e]‖₂. Physical arc-length positions along
      the undeformed centerline; guarantees ν = 1 in the reference config.
    directors_0: Array of shape (num_nodes, num_vec=3, num_dim=3) of orthonormal rest
      reference frames. Row 0 is d₁, row 1 is d₂, row 2 is d₃. Represents the stress-
      free undeformed state. The local material frame {d₁, d₂, d₃} at each node is a
      right-handed orthonormal triad: d₁ × d₂ = d₃. d₃ is the unit tangent along the rod
      d₁ and d₂ span the cross-section.
    fibers: `Fibers` dataclass with internal fiber geometry.
    num_translational_dofs_per_node: Number of translational dofs per node
      (default 3 for 3D).
    num_rotational_dofs_per_node: Number of rotational dofs per node (default 3).
    bounding_box: `BoundingBox` defining spatial limits of the rod.
    elem_lengths: Array of shape (num_elems,) of reference chord lengths.
    elem_arc_lengths: Array of shape (num_elems,) of Δŝₑ = arc_coords[e+1]
      − arc_coords[e]. Equal to element chord length; ensures ν = 1 in the
      reference configuration.
    elem_centers: Array of shape (num_elems, num_dim=3) of element midpoint
      global coordinates.
    elem_areas: Array of shape (num_elems,) of net cross-sectional areas,
      summed over all fibers.
    contact_radii: Array of shape (num_elems,) of contact radii for each element. Here,
      we assume the fibers are circular in cross-section and take into account the
      offset of the fibers from the element center. This is the maximum radius. We
      do not take into account the variation in the contact radius across the perimeter.
      We assume the cross-section is circular and computes the largest distance from the
      centerline to the edge of the cross-section. For non-circular cross-sections, the
      contact radius is an approximation and may not be accurate. This can be addressed
      by using a more sophisticated method for computing the contact radius.
    elem_volume: Array of shape (num_elems,) of reference element volumes.
    domain_volume: Scalar total reference volume of the rod.
    elem_moi_tensor: Array of shape (num_elems, 3, 3) of the second area moment of
      inertia tensor I_αβ = ∫ r_α r_β dA in the local material frame {d₁, d₂, d₃}.
      Assembled from the fiber offsets plus each fiber's `centroid_moi` via the
      parallel axis theorem. Entry [2, 2] is the polar second moment J = I₁₁ + I₂₂,
      which equals the torsion constant only for solid circular sections.
    elem_nodes: Integer array of shape (num_elems, nodes_per_elem=2) of global node
      numbers bounding each element.
    elem_node_coords: Array of shape (num_elems, nodes_per_elem=2, num_dim=3)
      of global Cartesian node pair coordinates for each element.
    elem_dof_mat: Integer array of shape (num_elems, num_dofs_per_element)
      of global dof numbers for each element.
    iK: Integer array of global row-scatter indices for stiffness assembly.
    jK: Integer array of global column-scatter indices for stiffness assembly.
    ref_kinematics: `CosseratKinematics` dataclass with `strain` [η₁, η₂, ν] and
      `curvature` [κ₁, κ₂, κ₃] both of shape (num_elems, 3) at the stress-free reference
      configuration. The stresses computed must discount these reference strains to
      avoid spurious pre-stresses in curved or pre-twisted reference geometries.
  """

  node_coords: jnp.ndarray
  arc_coords: jnp.ndarray
  directors_0: jnp.ndarray
  fibers: Fibers
  num_translational_dofs_per_node: int = 3
  num_rotational_dofs_per_node: int = 3

  bounding_box: BoundingBox = dataclasses.field(init=False)
  elem_lengths: jnp.ndarray = dataclasses.field(init=False)
  elem_arc_lengths: jnp.ndarray = dataclasses.field(init=False)
  elem_centers: jnp.ndarray = dataclasses.field(init=False)
  elem_areas: jnp.ndarray = dataclasses.field(init=False)
  contact_radii: jnp.ndarray = dataclasses.field(init=False)
  elem_volume: jnp.ndarray = dataclasses.field(init=False)
  domain_volume: jnp.ndarray = dataclasses.field(init=False)
  elem_moi_tensor: jnp.ndarray = dataclasses.field(init=False)
  elem_nodes: np.ndarray = dataclasses.field(init=False)
  elem_node_coords: jnp.ndarray = dataclasses.field(init=False)
  elem_dof_mat: np.ndarray = dataclasses.field(init=False)
  iK: np.ndarray = dataclasses.field(init=False)
  jK: np.ndarray = dataclasses.field(init=False)
  ref_kinematics: CosseratKinematics = dataclasses.field(init=False)

  @property
  def num_dim(self) -> int:
    """Returns the spatial dimension count of the Euclidean space (always 3)."""
    return 3

  @property
  def num_nodes_per_element(self) -> int:
    """Returns the count of nodes bounding each element segment."""
    return 2

  def __post_init__(self) -> None:
    """Computes static finite element connectivities and reference metrics."""

    self.num_nodes = self.node_coords.shape[0]
    self.num_elems = self.num_nodes - 1
    self.num_dofs_per_node = (
      self.num_translational_dofs_per_node + self.num_rotational_dofs_per_node
    )
    self.num_dofs = self.num_dofs_per_node * self.num_nodes
    self.num_dofs_per_element = self.num_dofs_per_node * self.num_nodes_per_element

    # (e)lement, (f)iber

    # Topological node connectivity lookup mapping
    element_indices = np.arange(self.num_elems)
    self.elem_nodes = np.stack([element_indices, element_indices + 1], axis=1)

    # Coordinates, midpoints, and net reference area properties
    self.elem_node_coords = jnp.stack(
      [self.node_coords[:-1, :], self.node_coords[1:, :]], axis=1
    )
    self.elem_centers = 0.5 * (self.node_coords[:-1] + self.node_coords[1:])
    self.elem_areas = jnp.sum(self.fibers.areas, axis=1)

    fiber_offset_norms = jnp.linalg.norm(self.fibers.offsets, axis=-1)  # (e, f)
    fiber_radii = fiber_offset_norms + jnp.sqrt(self.fibers.areas / jnp.pi)  # (e, f)
    self.contact_radii = jnp.max(fiber_radii, axis=-1)  # (e,)  maximum over fibers [m]

    # Reference segment volume and integrated domain metrics.
    chords = self.node_coords[1:] - self.node_coords[:-1]
    self.elem_lengths = jnp.linalg.norm(chords, axis=1)
    self.elem_arc_lengths = self.arc_coords[1:] - self.arc_coords[:-1]
    self.elem_volume = self.elem_areas * self.elem_arc_lengths
    self.domain_volume = jnp.sum(self.elem_volume)

    # Second area moment of inertia tensor I_αβ = ∫ r_α r_β dA, α,β ∈ {d₁, d₂}.
    # I₁₁ = ∫y₁²dA is the MOI about d₂ and pairs with κ₂; I₂₂ = ∫y₂²dA is the MOI
    # about d₁ and pairs with κ₁. The bending energy density is therefore
    #   ½E (κ₂² I₁₁ − 2 κ₁ κ₂ I₁₂ + κ₁² I₂₂),
    # i.e. the diagonal pairing is crossed and the I₁₂ term carries a minus sign.
    d1_offsets = self.fibers.offsets[..., _utils.Director.D1]  # (e, f)
    d2_offsets = self.fibers.offsets[..., _utils.Director.D2]  # (e, f)

    fib_I_d1 = jnp.einsum("ef -> e", self.fibers.centroid_moi[..., _utils.Director.D1])
    fib_I_d2 = jnp.einsum("ef -> e", self.fibers.centroid_moi[..., _utils.Director.D2])

    # Parallel axis thm: MOI about the d₂ axis adds to the d₁-offset term. (sim for d₂)
    i_11 = jnp.einsum("ef, ef -> e", self.fibers.areas, d1_offsets**2) + fib_I_d2
    i_22 = jnp.einsum("ef, ef -> e", self.fibers.areas, d2_offsets**2) + fib_I_d1
    i_12 = jnp.einsum("ef, ef -> e", self.fibers.areas, d1_offsets * d2_offsets)

    polar_j = i_11 + i_22
    zeros = jnp.zeros(self.num_elems)

    self.elem_moi_tensor = jnp.stack(
      [
        jnp.stack([i_11, i_12, zeros], axis=-1),  # row 0: [I₁₁, I₁₂, 0]
        jnp.stack([i_12, i_22, zeros], axis=-1),  # row 1: [I₁₂, I₂₂, 0]
        jnp.stack([zeros, zeros, polar_j], axis=-1),  # row 2: [0,   0,  J]
      ],
      axis=1,
    )  # (e, 3, 3)

    # Assemble global Degree-of-Freedom maps sequentially on host
    elem_dof_list = []
    for e in range(self.num_elems):
      left_node, right_node = e, e + 1
      left_dofs = [
        left_node * self.num_dofs_per_node + j for j in range(self.num_dofs_per_node)
      ]
      right_dofs = [
        right_node * self.num_dofs_per_node + j for j in range(self.num_dofs_per_node)
      ]
      elem_dof_list.append(left_dofs + right_dofs)

    self.elem_dof_mat = np.array(elem_dof_list, dtype=np.int32)

    # Construct Kronecker row/column index vectors for global matrix mapping
    self.iK = np.repeat(self.elem_dof_mat, self.num_dofs_per_element, axis=1).flatten()
    self.jK = np.tile(self.elem_dof_mat, (1, self.num_dofs_per_element)).flatten()

    # Strain measures at the stress-free reference configuration.
    self.ref_kinematics = compute_discrete_cosserat_kinematics(
      node_coords=self.node_coords,
      directors=self.directors_0,
      arc_coords=self.arc_coords,
    )

    # Compute the spatial bounding box from centerline coordinates
    min_xyz = jnp.min(self.node_coords, axis=0)
    max_xyz = jnp.max(self.node_coords, axis=0)
    self.bounding_box = BoundingBox(
      x=_utils.Extent(min=float(min_xyz[0]), max=float(max_xyz[0])),
      y=_utils.Extent(min=float(min_xyz[1]), max=float(max_xyz[1])),
      z=_utils.Extent(min=float(min_xyz[2]), max=float(max_xyz[2])),
    )

  def get_dof_indices(
    self,
    dof_enum: _utils.Translate_or_Rotate,
    node_indices: np.ndarray,
  ) -> np.ndarray:
    """Returns the global dof index of a specific component at each target node.

    Each node stores dofs packed as:

            [ trans_0, ..., trans_{t−1},  rot_0, ..., rot_{r−1} ]
             ←── indices 0 … t−1 ───→  ←── indices t … t+r−1 ──→

    where t = `num_translational_dofs_per_node` and r = `num_rotational_dofs_per_node`.
    The global dof index for component `local_dof_index` at node i is:

      global_dof[i] = node_indices[i] * num_dofs_per_node + local_dof_index

    Args:
      dof_enum: A `_utils.Translation` (e.g., `Translation.X`) or `_utils.Rotation`
         (e.g., `Rotation.THETA_Z`) identifying the dof component.
      node_indices: Integer array of shape (num_target_nodes,) containing
        global node indices at which to retrieve dof positions.

    Returns:
      dof_indices: Integer array of shape (num_target_nodes,) of global dof indices for
        the requested component at each target node.
    """
    if isinstance(dof_enum, _utils.Translation):
      local_dof_index = dof_enum.value
    else:
      local_dof_index = self.num_translational_dofs_per_node + dof_enum.value

    return (node_indices * self.num_dofs_per_node + local_dof_index).astype(np.int32)

  def wrap_rotational_dofs(
    self,
    u: jnp.ndarray,
  ) -> jnp.ndarray:
    """Reduce every nodal rotation in a displacement vector to ‖θ‖ ≤ π.

    Applies `utils.wrap_rotation_vector` to the rotational block of each node and
    leaves the translational block untouched. The encoded rotations are unchanged,
    so the returned vector describes the identical configuration.

    Args:
      u: Array of shape `(num_dofs,)` [m, rad] of nodal displacements and rotations.

    Returns:
      Array of shape `(num_dofs,)` with the rotational entries wrapped.
    """
    num_trans = self.num_translational_dofs_per_node
    nodal = u.reshape(self.num_nodes, self.num_dofs_per_node)
    wrapped = jax.vmap(_utils.wrap_rotation_vector)(nodal[:, num_trans:])
    return nodal.at[:, num_trans:].set(wrapped).reshape(-1)


def _compute_centerline(
  analytical_curve: _FLOAT_TO_ARRAY_FN,
  seed_binormal: jnp.ndarray,
  parametric_coords: Optional[jnp.ndarray] = None,
  imperfection_amplitude: float = 0.0,
  initial_bow_ratio: float = 0.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Generates spatial node coordinates and cumulative arc-length coordinates.

  The `arc_coords` is computed as cumulative Euclidean chord lengths :

                            Δŝₑ = ‖r(sₑ₊₁) − r(sₑ)‖₂

  so the axial stretch ν = lₑ / Δŝₑ = 1.0 at the reference state for every
  element, regardless of the parametric speed ‖dr/ds‖.

  Args:
    analytical_curve: Func mapping scalar param s -> position vec of shape (num_dim=3,).
    parametric_coords: Array of shape (num_nodes,) of parameter coordinates at which to
      evaluate the curve.
    imperfection_amplitude: @see `build_rod_geometry`.
    initial_bow_ratio: @see `build_rod_geometry`

  Returns:
    A tuple containing:
      node_coords: Array of shape (num_nodes, num_dim=3) of node coordinates.
      arc_coords: Array of shape (num_nodes,) of cumulative Euclidean chord-lengths
        coordinates, with arc_coords[0] = 0.0 and
        arc_coords[e+1] − arc_coords[e] = ‖node_coords[e+1] − node_coords[e]‖₂.
  """
  # num_(n)odes, num_(d)im, num_(e)lements
  nodes = jax.vmap(analytical_curve)(parametric_coords)  # (n, d)
  nodes = nodes + imperfection_amplitude * jnp.array(
    np.random.default_rng(seed=0).standard_normal(nodes.shape)
  )

  chord_lengths = jnp.linalg.norm(nodes[1:] - nodes[:-1], axis=1)  # (e)
  arc_coords = jnp.concatenate([jnp.zeros(1), jnp.cumsum(chord_lengths)])  # (n)

  total_length = arc_coords[-1]
  bow = initial_bow_ratio * total_length * jnp.sin(jnp.pi * arc_coords / total_length)
  nodes = nodes + jnp.einsum("n, d -> nd", bow, seed_binormal)

  return nodes, arc_coords


def _compute_reference_frames(
  node_coords: jnp.ndarray,
  intrinsic_twist: jnp.ndarray,
  seed_binormal: jnp.ndarray,
) -> jnp.ndarray:
  """Compute nodal orthonormal material frames via Bishop parallel transport.

  Propagates a right-handed frame {d₁, d₂, d₃} along the rod from node 0 to
  node N via the minimum-rotation (Bishop) rule, then applies intrinsic twist.
  The frame satisfies d₁ × d₂ = d₃ at every node by construction.

  The seed frame at node 0 is constructed from `seed_binormal`, the desired
  physical direction of d₂ at the rod start (the out-of-plane binormal):

      d₁₀ = (seed_binormal × d₃₀) / ‖seed_binormal × d₃₀‖
      d₂₀ = d₃₀ × d₁₀  (≡ seed_binormal when seed_binormal ⊥ d₃₀)

  Correct choices for common rod planes:
    rod in x-y plane → seed_binormal = [0, 0, 1]  → d₁ = in-plane transverse
    rod in x-z plane → seed_binormal = [0, 1, 0]  → d₁ = in-plane transverse
    rod in y-z plane → seed_binormal = [1, 0, 0]  → d₁ = in-plane transverse

  For a fully 3D rod, seed_binormal seeds d₂ at node 0 and Bishop transport
  propagates the frame consistently along the curve.

  Algorithm:
    1. Chord tangents: t_e = (r_{e+1} − r_e) / ‖r_{e+1} − r_e‖, shape (E, 3).
    2. Node tangents: endpoint nodes take the adjacent chord tangent directly.
       Interior node n uses normalize(t_{n−1} + t_n).
    3. Seed: d₁₀ = normalize(seed_binormal × d₃₀).
    4. Bishop transport node n → n+1 via compute_rodrigues_matrix(θ):
           θ = (φ / sinφ) · (t_n × t_{n+1}),   φ = arccos(t_n · t_{n+1})
           d₁_{n+1} = R(θ) d₁_n  (renormalised to correct float drift)
       Taylor branch for φ² < 1e-12: φ/sinφ ≈ 1 + φ²/6 (handled internally
       by `compute_rodrigues_matrix`).
    5. Bishop binormal: d₂_n = d₃_n × d₁_n at each node.
    6. Apply intrinsic twist θ(s) as a rotation about d₃:
           d₁ =  cosθ · d₁_bishop + sinθ · d₂_bishop
           d₂ = −sinθ · d₁_bishop + cosθ · d₂_bishop

  Args:
    node_coords: Array of shape (num_nodes, 3). Nodal positions in the global
      Cartesian frame [m].
    intrinsic_twist: Array of shape (num_nodes,). Intrinsic twist angle [rad]
      at each node, rotating the Bishop frame about d₃ after transport.
    seed_binormal: Array of shape (3,). Global-frame direction seeding d₂ at
      node 0. Must not be parallel to d₃₀. Determines which physical direction
      d₁ faces and is propagated consistently by Bishop transport.

  Returns:
    directors: Array of shape (num_nodes, 3, 3). `directors[n, k, :]` is the
      k-th material director at node n in global coordinates (k=0 → d₁,
      k=1 → d₂, k=2 → d₃). Rows are orthonormal, det = +1 at every node.
  """
  # --- Steps 1 & 2: chord tangents → smooth node tangents ---
  chord_vecs = node_coords[1:] - node_coords[:-1]  # (e, 3)
  chord_tangents = _utils.safe_normalize(chord_vecs, axis=-1)

  # Interior nodes average the two adjacent chord tangents before normalising.
  interior_tangents_unnorm = chord_tangents[:-1] + chord_tangents[1:]  # (e-1, 3)
  node_tangents_unnorm = jnp.concatenate(
    [chord_tangents[:1], interior_tangents_unnorm, chord_tangents[-1:]], axis=0
  )  # (n, 3)
  node_tangents = _utils.safe_normalize(node_tangents_unnorm, axis=-1)  # (n, 3)

  # --- Step 3: seed Bishop normal at node 0 ---
  # d₁₀ = (seed_binormal × d₃₀) / ‖...‖;  seed_binormal ≈ d₂₀ when ⊥ d₃₀.
  seed_bishop_normal = jnp.cross(seed_binormal, node_tangents[0])
  seed_bishop_normal = seed_bishop_normal / jnp.linalg.norm(seed_bishop_normal)

  # --- Step 4: Bishop parallel transport ---
  def _transport_step(
    carry: Tuple[jnp.ndarray, jnp.ndarray],
    curr_tangent: jnp.ndarray,
  ) -> Tuple[Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    prev_bishop_normal, prev_tangent = carry

    # Minimum-rotation vector θ mapping prev_tangent → curr_tangent.
    # |cross(t, t')| = sinφ,  dot(t, t') = cosφ  →  θ = (φ/sinφ) · cross
    rotation_axis = jnp.cross(prev_tangent, curr_tangent)  # sinφ · n̂
    cos_phi = jnp.clip(jnp.dot(prev_tangent, curr_tangent), -1.0 + 1e-12, 1.0 - 1e-12)
    phi = jnp.arccos(cos_phi)
    phi2 = phi * phi
    small = phi2 < 1e-12
    safe_sin = jnp.where(small, 1.0, jnp.sin(phi))
    # φ/sinφ Taylor branch: 1 + φ²/6 + O(φ⁴).
    scale = jnp.where(small, 1.0 + phi2 / 6.0, phi / safe_sin)

    transport_rotation = _utils.compute_rodrigues_matrix(scale * rotation_axis)
    next_bishop_normal = jnp.einsum("ij,j->i", transport_rotation, prev_bishop_normal)
    # Renormalise to prevent floating-point magnitude drift.
    next_bishop_normal = next_bishop_normal / jnp.linalg.norm(next_bishop_normal)
    return (next_bishop_normal, curr_tangent), next_bishop_normal

  _, transported_bishop_normals = jax.lax.scan(
    f=_transport_step,
    init=(seed_bishop_normal, node_tangents[0]),
    xs=node_tangents[1:],
  )  # (num_nodes-1, 3)

  bishop_normals = jnp.concatenate(
    [seed_bishop_normal[None], transported_bishop_normals], axis=0
  )  # (num_nodes, 3)

  # --- Steps 5 & 6: bishop binormals + intrinsic twist ---
  # bishop_binormals = d₃ × d₁_bishop; completes the right-handed Bishop frame.
  bishop_binormals = jnp.cross(node_tangents, bishop_normals)  # (num_nodes, 3)

  cos_tw = jnp.cos(intrinsic_twist)  # (num_nodes,)
  sin_tw = jnp.sin(intrinsic_twist)  # (num_nodes,)

  d1 = jnp.einsum("n,ni->ni", cos_tw, bishop_normals) + jnp.einsum(
    "n,ni->ni", sin_tw, bishop_binormals
  )
  d2 = jnp.einsum("n,ni->ni", -sin_tw, bishop_normals) + jnp.einsum(
    "n,ni->ni", cos_tw, bishop_binormals
  )
  return jnp.stack([d1, d2, node_tangents], axis=1)  # (num_nodes, 3, 3)


def build_rod_geometry(
  analytical_curve: _FLOAT_TO_ARRAY_FN,
  parametric_coords: jnp.ndarray,
  fiber_offsets: Union[jnp.ndarray, _FLOAT_TO_ARRAY_FN],
  fiber_areas: Union[jnp.ndarray, _FLOAT_TO_ARRAY_FN],
  fiber_mass_density: Union[jnp.ndarray, _FLOAT_TO_ARRAY_FN],
  fiber_centroid_moi: Union[jnp.ndarray, _FLOAT_TO_ARRAY_FN],
  seed_binormal: jnp.ndarray,
  fiber_winding_angles: Optional[Union[jnp.ndarray, _FLOAT_TO_ARRAY_FN]] = None,
  intrinsic_twist: Union[jnp.ndarray, Callable[[float], float]] = lambda s: 0.0,
  num_translational_dofs_per_node: int = 3,
  num_rotational_dofs_per_node: int = 3,
  imperfection_amplitude: float = 0.0,
  initial_bow_ratio: float = 0.0,
) -> Rod:
  """Initializes a `Rod` from a centerline and fiber geometry.

  Evaluates cross-sectional properties across elements and transforms local
  fiber profiles into global 3D space via explicit parallel tensor contractions.

  Fiber Spatial Position:
    r_fiber[e, f, i] = r_center[e, i]
                     + offsets[e, f, 0] · d̂₁_mean[e, i]
                     + offsets[e, f, 1] · d̂₂_mean[e, i]

  where d̂₁_mean and d̂₂_mean are the d₁ and d₂ columns of the SO(3) geodesic
  mean frame at the element midpoint.

  Args:
    analytical_curve: Func mapping scalar parametric coordinate k -> coordiante on the
      curve of shape (num_dim=3,).
    parametric_coords: Array of shape (num_nodes,) of parameteric coordinates at which
      to evaluate the curve.
    fiber_offsets: Array of shape (num_elems, num_fibers, num_crossec_dim=2) or callable
      mapping arc length s → (num_fibers, 2). Local material-frame offsets of each fiber
      from the centerline. Index 0 is the d₁ offset. Index 1 is the d₂ offset.
    fiber_areas: Array of shape (num_elems, num_fibers) or a function mapping arc length
      coordinate to fiber cross-sectional areas F: s -> (num_fibers,).
    fiber_mass_density: Array of shape (num_elems, num_fibers) or function mapping arc
      length coordinate to the mass densities of all the fibers F: s -> (num_fibers,).
    fiber_centroid_moi: Array of shape (num_elems, num_fibers, num_crossec_dim=2) or
      function mapping arc length coordinate to the fiber centroid second moment of
      inertia: s -> (num_fibers, num_crossec_dim=2).
    seed_binormal: Array of shape (num_dim=3,). Global-frame direction that seeds the
      binormal d₂ at the rod start node. For planar rods pass the normal to the
      bending plane (e.g. [0,0,1] for a rod in the x-y plane). This determines
      which physical direction d₁ faces.
    intrinsic_twist: Array of shape (num_nodes,) or callable theta(s) where `s` is the
      arc-length coordinate and `theta` is the intrinsic twist angle in radians. When
      provided as an array, `intrinsic_twist[n]` gives the twist angle at node n. When
      provided as a callable, it is evaluated at each `arc_coords[n]` to give the twist
      at each node. The default is a zero intrinsic twist.
    num_translational_dofs_per_node: Translational dofs per node. Default 3.
    num_rotational_dofs_per_node: Rotational dofs per node. Default 3 for 3D;
      use 1 for a planar 2D Cosserat rod.
    imperfection_amplitude: Scalar isotropic random Gaussian jitter added to node
      coordinates. Default 0.0 (no perturbation). Use 0.001–0.01 × rod cross-section
      radius to break bifurcation symmetry for buckling analyses. Randomness seed of
      0 is used for reproducibility.
    initial_bow_ratio: Amplitude of a deterministic half-wave transverse bow, as a
      fraction of the total arc length. Breaks the symmetry of an axially loaded
      straight rod, which otherwise sits on a bifurcation with a singular tangent.
      Values around 1e-3 are typical. Zero leaves the centerline untouched.

  Returns:
    rod: Fully populated `Rod` instance with all geometric layouts, material
      properties, bounding box, and finite element connectivity matrices.
  """
  # (e)lement = (n)odes - 1, (f)iber, (c)ross_sec_dim = 2, (d)im = 3,
  node_coords, arc_coords = _compute_centerline(
    analytical_curve=analytical_curve,
    seed_binormal=seed_binormal,
    parametric_coords=parametric_coords,
    imperfection_amplitude=imperfection_amplitude,
    initial_bow_ratio=initial_bow_ratio,
  )  # (n, d), (n,)

  twist_angles = intrinsic_twist  # (n,) if precomputed array
  if callable(intrinsic_twist):
    twist_angles = jax.vmap(intrinsic_twist)(arc_coords)

  directors_0 = _compute_reference_frames(
    node_coords=node_coords,
    intrinsic_twist=twist_angles,
    seed_binormal=seed_binormal,
  )  # (n, d, d)

  elem_center_coords = 0.5 * (node_coords[:-1] + node_coords[1:])  # (e, d)
  mean_directors = jax.vmap(_utils.interpolate_directors_geodesic_so3)(
    directors_0[:-1, :, :], directors_0[1:]
  )  # (e, d, d)  rows are d₁, d₂, d₃
  cross_directors = mean_directors[:, :2, :]  # (e, c, d)  d₁ and d₂ rows

  arc_length_midpoints = 0.5 * (arc_coords[:-1] + arc_coords[1:])  # (e,)

  offsets = fiber_offsets  # (e, f, c) if precomputed array
  if callable(fiber_offsets):
    offsets = jax.vmap(fiber_offsets)(arc_length_midpoints)  # (e, f, c)

  areas = fiber_areas  # (e, f) if precomputed array
  if callable(fiber_areas):
    areas = jax.vmap(fiber_areas)(arc_length_midpoints)  # (e, f)

  densities = fiber_mass_density
  if callable(fiber_mass_density):
    densities = jax.vmap(fiber_mass_density)(arc_length_midpoints)  # (e, f)

  centroid_moi = fiber_centroid_moi  # (e, f, c) if precomputed array
  if callable(fiber_centroid_moi):
    centroid_moi = jax.vmap(fiber_centroid_moi)(arc_length_midpoints)  # (e, f, c)

  winding_angles = fiber_winding_angles  # (e, f) if precomputed array
  if fiber_winding_angles is None:
    winding_angles = jnp.zeros(offsets.shape[:2])  # (e, f)
  elif callable(fiber_winding_angles):
    winding_angles = jax.vmap(fiber_winding_angles)(arc_length_midpoints)  # (e, f)

  winding_directions = compute_fiber_winding_directions(
    fiber_offsets=offsets,
    winding_angles=winding_angles,
  )  # (e, f, d)

  displacements = jnp.einsum("efc, ecd -> efd", offsets, cross_directors)
  fiber_positions = elem_center_coords[:, jnp.newaxis, :] + displacements

  fiber_bundle = Fibers(
    offsets=offsets,
    areas=areas,
    winding_angles=winding_angles,
    winding_directions=winding_directions,
    positions=fiber_positions,
    centroid_moi=centroid_moi,
    fiber_mass_density=densities,
  )

  return Rod(
    node_coords=node_coords,
    arc_coords=arc_coords,
    directors_0=directors_0,
    fibers=fiber_bundle,
    num_translational_dofs_per_node=num_translational_dofs_per_node,
    num_rotational_dofs_per_node=num_rotational_dofs_per_node,
  )


def compute_curvature_from_centerline(
  node_coords: jnp.ndarray,
  seed_binormal: jnp.ndarray,
  intrinsic_twist: Optional[jnp.ndarray] = None,
) -> CosseratKinematics:
  """Compute the Darboux vector of a discrete space curve.

  Builds the nodal material frames by Bishop parallel transport along the polyline,
  applies any prescribed intrinsic twist about the tangent, and evaluates the
  discrete Cosserat kinematics against the cumulative chord length.

  Args:
    node_coords: Array of shape `(num_nodes, num_dim=3)` [m] of points along the
      curve, ordered from one end to the other. Consecutive points must be distinct,
      since the tangent is formed by normalising the chord between them.
    seed_binormal: Array of shape `(num_dim=3,)` fixing the orientation of the frame
      at the first node. Must not be parallel to the initial chord tangent, which
      would leave the transverse directors undetermined.
    intrinsic_twist: Optional array of shape `(num_nodes,)` [rad] of the twist angle
      about the tangent at each node, applied on top of the Bishop frame. Defaults
      to zero at every node.

  Returns:
    `CosseratKinematics` whose `strain` has shape `(num_elems, num_components=3)`
      and whose `curvature` has shape `(num_elems, num_components=3)` [1/m].
  """
  if intrinsic_twist is None:
    intrinsic_twist = jnp.zeros(node_coords.shape[0])

  chord_lengths = jnp.linalg.norm(jnp.diff(node_coords, axis=0), axis=-1)
  arc_coords = jnp.concatenate([jnp.zeros(1), jnp.cumsum(chord_lengths)])
  directors = _compute_reference_frames(
    node_coords=node_coords,
    intrinsic_twist=intrinsic_twist,
    seed_binormal=seed_binormal,
  )
  return compute_discrete_cosserat_kinematics(
    node_coords=node_coords, directors=directors, arc_coords=arc_coords
  )


def extract_coords_and_director_from_deformation(
  rod: Rod,
  u: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Extract deformed node coords, director frames and fiber coords from deformation.

  Unpacks global displacement vector into its translational and rotational parts
  using the rod's own DOF layout, rotates the reference directors by the nodal
  rotation vectors, and carries the fiber offsets into the deformed configuration.
  A two-dimensional `u` is read as a load-step history and mapped state by state, so
  every output gains the same leading axis.

  Args:
    rod: `Rod` supplying the reference configuration and DOF layout.
    u: Array of shape `(num_dofs,)` or `(num_load_steps, num_dofs)` [m, rad] of nodal
      displacements and rotations.

  Returns:
    A tuple (each array gains a leading `num_load_steps` axis when `u` is 2-dim)
      containing:
      - node_coords : Array of shape (num_nodes, num_dim=3) of the coordinates of the
          deformed centerline.
      - directors: Array of shape (num_nodes, num_dir=3, num_dim=3) of the rotated
          directors at the nodes of the rod.
      - fiber_coordinates: Array of shape (num_nodes, num_fibers, num_dim=3) of the
          coordinates of the fibers upon deformation.
  """
  if u.ndim == 2:
    single_state = ft.partial(extract_coords_and_director_from_deformation, rod)
    return jax.vmap(single_state)(u)

  num_trans = rod.num_translational_dofs_per_node
  nodal = u.reshape(rod.num_nodes, rod.num_dofs_per_node)
  displacements, rotations = nodal[:, :num_trans], nodal[:, num_trans:]

  directors = jax.vmap(_utils.rotate_directors)(rotations, rod.directors_0)
  fiber_coordinates = compute_fiber_positions(
    node_coords_ref=rod.node_coords,
    directors_ref=rod.directors_0,
    fiber_offsets=rod.fibers.nodal_offsets(rod.elem_arc_lengths),
    node_displacements=displacements,
    incremental_rotations=rotations,
  )
  return rod.node_coords + displacements, directors, fiber_coordinates
