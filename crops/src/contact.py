"""Penalty-based contact mechanics for Cosserat rod systems.

Implements both self-contact and environmental contacts.

For self-contact, the rod computes a soft penalty force based on the distance of
one element of a rod to every other (leaving out self-interactions and nearby elements).
We also use a block diagonal approximation (similar to lumped mass matrices) to reduce
the computational cost of the contact force calculations.

For environmental contacts, the rod interacts with a fixed obstacle or wall, computing a
soft penalty force based on the distance to the obstacle.
The obstacle geometry is described by an `Obstacle` instance whose signed distance
function φ : ℝⁿ → ℝ follows the convention:

            φ(x) > 0  →  free space (no contact)
            φ(x) = 0  →  wall surface
            φ(x) < 0  →  inside wall (penetration)

`Obstacle` provides continuous, differentiable cubic spline interpolation over a
background SDF grid. The contact normal ∇φ is obtained exactly via applying autodiff.
"""

import dataclasses
import functools as ft
import warnings
from typing import Literal, Optional

import interpax
import jax
import jax.experimental.sparse as jax_sprs
import jax.numpy as jnp
import numpy as np
import pysdf
import skfmm
import trimesh

import crops.src.mesher as _mesher
import crops.src.utils as _utils

# Relative tolerance on the Gram determinant below which two segments are treated as
# parallel. Scaled by len_a * len_b so it is independent of the element size.
_PARALLEL_DET_TOL = 1e-12


def _segment_segment_distance(
  p0: jnp.ndarray,
  p1: jnp.ndarray,
  q0: jnp.ndarray,
  q1: jnp.ndarray,
) -> jnp.ndarray:
  """Minimum Euclidean distance between two finite line segments in ℝⁿ.

  For details, see:
    David Eberly, Robust Computation of Distance Between Line Segments.

  Implements the branchless five-candidate Eberly formula. Segments A and B
  are parameterised as:

                    P(s) = p0 + s · (p1 − p0),   s ∈ [0, 1]
                    Q(t) = q0 + t · (q1 − q0),   t ∈ [0, 1]

  The gap vector at (s, t) is:

                    gap(s, t) = P(s) − Q(t)
                              = (p0 − q0) + s · seg_a − t · seg_b

  where `seg_a = p1 − p0` and `seg_b = q1 − q0`. The objective `f(s, t) = |gap|²` is a
  quadratic on [0, 1]², fully determined by five scalar inner products:

                len_sq_a   = seg_a · seg_a      squared length of segment A
                len_sq_b   = seg_b · seg_b      squared length of segment B
                dot_segs   = seg_a · seg_b      alignment between segment directions
                gap_proj_a = gap_origin · seg_a  component of origin gap along seg A
                gap_proj_b = gap_origin · seg_b  component of origin gap along seg B

  where `gap_origin = p0 − q0`. Setting ∂f/∂s = 0 and ∂f/∂t = 0 gives the 2×2 linear
  system for the unconstrained interior minimizer (s*, t*):

                  [ len_sq_a   −dot_segs ] [s*]   [−gap_proj_a]
                  [ −dot_segs   len_sq_b ] [t*] = [ gap_proj_b ]

  with determinant Δ = len_sq_a · len_sq_b − dot_segs². By Cauchy-Schwarz Δ ≥ 0; Δ = 0
  iff the segments are parallel, handled by a safe divisor.

  By Cramer's rule:

                    s* = (dot_segs · gap_proj_b − gap_proj_a · len_sq_b) / Δ
                    t* = (len_sq_a · gap_proj_b − gap_proj_a · dot_segs) / Δ

  If (s*, t*) lies outside [0, 1]², the constrained minimum is on one of
  the four edges. Each edge reduces to a one-variable quadratic whose
  unconstrained minimizer is:

                    edge s = 0:  t_opt = gap_proj_b / len_sq_b
                    edge s = 1:  t_opt = (gap_proj_b + dot_segs) / len_sq_b
                    edge t = 0:  s_opt = −gap_proj_a / len_sq_a
                    edge t = 1:  s_opt = (dot_segs − gap_proj_a) / len_sq_a

  Each is clamped to [0, 1]. This covers all four corners of the square as well (e.g.,
  corner (0, 0) is reached by the s=0 edge with t clamped down and by the t=0 edge with
  s clamped down). Evaluating |gap|² at all five candidates and taking `jnp.min` gives
  the globally correct minimum.

  Args:
    p0: Start of segment A, shape `(num_dim,)` [m].
    p1: End of segment A, shape `(num_dim,)` [m].
    q0: Start of segment B, shape `(num_dim,)` [m].
    q1: End of segment B, shape `(num_dim,)` [m].

  Returns:
    Scalar minimum distance d_min `()` [m]. Safe against d_min = 0 via an internal
      floor of 1e-30 before the square root.
  """
  seg_a = p1 - p0  # (num_dim,)
  seg_b = q1 - q0  # (num_dim,)
  gap_origin = p0 - q0  # (num_dim,)

  len_sq_a = jnp.einsum("i,i->", seg_a, seg_a)
  len_sq_b = jnp.einsum("i,i->", seg_b, seg_b)
  dot_segs = jnp.einsum("i,i->", seg_a, seg_b)
  gap_proj_a = jnp.einsum("i,i->", gap_origin, seg_a)
  gap_proj_b = jnp.einsum("i,i->", gap_origin, seg_b)

  det = len_sq_a * len_sq_b - dot_segs**2  # Δ ≥ 0 by Cauchy-Schwarz

  is_parallel = det <= _PARALLEL_DET_TOL * (len_sq_a * len_sq_b)
  is_degenerate_a = len_sq_a <= _PARALLEL_DET_TOL * len_sq_b
  is_degenerate_b = len_sq_b <= _PARALLEL_DET_TOL * len_sq_a

  safe_det = jnp.where(is_parallel, 1.0, det)
  safe_len_a = jnp.where(is_degenerate_a, 1.0, len_sq_a)
  safe_len_b = jnp.where(is_degenerate_b, 1.0, len_sq_b)

  s_unconstrained = jnp.where(
    is_parallel, 0.0, (dot_segs * gap_proj_b - gap_proj_a * len_sq_b) / safe_det
  )
  t_unconstrained = jnp.where(
    is_parallel, 0.0, (len_sq_a * gap_proj_b - gap_proj_a * dot_segs) / safe_det
  )

  # Unconstrained minimizer on each boundary edge, clamped to [0, 1]
  t_at_s0 = jnp.where(is_degenerate_b, 0.0, jnp.clip(gap_proj_b / safe_len_b, 0.0, 1.0))
  t_at_s1 = jnp.where(
    is_degenerate_b, 0.0, jnp.clip((gap_proj_b + dot_segs) / safe_len_b, 0.0, 1.0)
  )
  s_at_t0 = jnp.where(
    is_degenerate_a, 0.0, jnp.clip(-gap_proj_a / safe_len_a, 0.0, 1.0)
  )
  s_at_t1 = jnp.where(
    is_degenerate_a, 0.0, jnp.clip((dot_segs - gap_proj_a) / safe_len_a, 0.0, 1.0)
  )

  def _gap_norm_sq(s_val: jnp.ndarray, t_val: jnp.ndarray) -> jnp.ndarray:
    """Squared distance |gap(s, t)|² at one candidate (s, t)."""
    gap = gap_origin + s_val * seg_a - t_val * seg_b
    return jnp.einsum("i,i->", gap, gap)

  d_sq_interior = _gap_norm_sq(
    jnp.clip(s_unconstrained, 0.0, 1.0),
    jnp.clip(t_unconstrained, 0.0, 1.0),
  )
  d_sq_edge_s0 = _gap_norm_sq(0.0, t_at_s0)
  d_sq_edge_s1 = _gap_norm_sq(1.0, t_at_s1)
  d_sq_edge_t0 = _gap_norm_sq(s_at_t0, 0.0)
  d_sq_edge_t1 = _gap_norm_sq(s_at_t1, 1.0)

  d_min_sq = jnp.min(
    jnp.stack(
      [
        d_sq_interior,
        d_sq_edge_s0,
        d_sq_edge_s1,
        d_sq_edge_t0,
        d_sq_edge_t1,
      ]
    )
  )

  return jnp.sqrt(jnp.maximum(d_min_sq, 1e-12))


@dataclasses.dataclass
class SelfContact:
  """Container for self-contact topology and penalty parameters for a rod.

  Stores the precomputed (non-adjacent) element pair list and global DOF scatter indices
  needed to assemble self-contact contributions.

  Attributes:
    pairs: Valid (non-adjacent) element pairs, shape `(num_valid_pairs, 2)`. Each row
      (a, b)` satisfies `a ∈ [0, N − w)` and `b ∈ [a + w, N)`, where `N = num_elems` and
      `w = adjacency_exclusion`, so `b − a ≥ w` for all entries.
    pair_trans_dof_mat: Global translational dof index matrix for each element of each
      pair, of shape`(num_valid_pairs, 2, num_translational_dofs_per_element)`.
      `pair_trans_dof_mat[p, i, :]` holds the global translational dof indices for the
      ith (0 or 1) element of pair p. Directly analogous to `rod.elem_dof_mat` but
      indexed over pairs, constrained to the translational dofs of their two constituent
      elements rather than over all elements.
    num_dofs_per_element: The number of degrees of freedom per element. Encompasses the
      translational and rotational degrees of freedom from all nodes in the element.
    num_dofs_per_node: The number of degrees of freedom per node. Includes both
      translational and rotational degrees of freedom.
    num_translational_dofs_per_node: (n_t) The num of translational dofs per node.
    sc_trans_iK: Global row translational dof scatter indices for assembling the
      per-element lumped contact blocks, shape `(num_elems · (2*n_t)²,)`. Pair blocks
      are summed onto their elements before assembly, so this scales with the number
      of elements rather than the number of pairs.
    sc_trans_jK: Global col translational dof scatter idxs, similar to `sc_trans_iK`.
    stiffness: Penalty stiffness k_sc [N/m²]. The effective contact stiffness per pair
      scales as k_sc · L [N/m], where L is the element length. This must satisfy
      k_sc · L ≪ min(EA/L, EI/L³) so that the contact contribution is much smaller than
      the rod's minimum elastic stiffness, preventing Newton-Raphson overshoot at
      contact onset.
    smoothing_fraction: Fraction of the combined contact radius (r_a + r_b) over which
      penalty force transitions smoothly from zero to active. Sets the softplus
      transition width as smoothing_fraction · (r_a + r_b). Default 0.05.
    min_separation_fraction: Smallest segment separation, as a fraction of the combined
      contact radius, at which the penalty force is still evaluated. Below this the pair
      is treated as being at this separation.
    max_candidate_pairs: Number K of pairs passed from the broad phase to the exact
      contact evaluation. Each call keeps the K pairs whose bounding balls are closest
      and skips the rest, which contribute exactly zero as long as no more than K pairs
      are within contact range. A warning is raised when that is violated. `None`
      disables the broad phase and evaluates every pair.
  """

  pairs: np.ndarray
  pair_trans_dof_mat: np.ndarray
  num_dofs_per_element: int
  num_dofs_per_node: int
  num_translational_dofs_per_node: int
  sc_trans_iK: np.ndarray
  sc_trans_jK: np.ndarray
  stiffness: float
  smoothing_fraction: float = 0.05
  min_separation_fraction: float = 0.01
  max_candidate_pairs: Optional[int] = None

  @property
  def num_contact_pairs(self) -> int:
    """Returns the total number of valid non-adjacent element pairs."""
    return self.pairs.shape[0]

  @classmethod
  def from_rod(
    cls,
    rod: _mesher.Rod,
    stiffness: float,
    adjacency_exclusion: int = 2,
    smoothing_fraction: float = 0.05,
    max_candidate_pairs: Optional[int] = None,
  ) -> "SelfContact":
    """Constructs a `SelfContact` from a `Rod` mesh topology.

    Enumerates all element pairs with index gap `b − a ≥ adjacency_exclusion` and
    precomputes global dof scatter indices. Elements with gap < 2 are always excluded.
    gap 1 means shared node (endpoints coincide, d_min = 0, spurious infinite force).
    The default exclusion of 2 additionally skips the nearest non-touching neighbours.

    Args:
      rod: The `Rod` instance.
      stiffness: Penalty stiffness k_sc [N/m²]. The contact penalty addsa soft spring
        between pairs that overlap with effective stiffness of k_sc · elem_length. This
        must be less than the bending stiffness (~ EI / L_e³) of the rod element.
      adjacency_exclusion: Minimum element index gap `b − a` for a pair to be included.
        Must be ≥ 2. Default 2.
      smoothing_fraction: Softplus transition width as a fraction of the combined
        contact radius. Default 0.05.
      max_candidate_pairs: Broad-phase capacity K, see `SelfContact`. Values at or
        above the number of valid pairs disable the broad phase. Default `None`.

    Returns:
      The `SelfContact` instance.
    """
    # num_(e)lems, num_(v)alid_pairs, num_(d)ofs_per_elem, (n)odes_per_elem=2, (p)air=2
    num_elems = rod.num_elems
    n_dn = rod.num_dofs_per_node
    n_t = rod.num_translational_dofs_per_node

    mean_le = float(rod.elem_arc_lengths.mean())
    max_2r = float(2.0 * rod.contact_radii.max())
    if (adjacency_exclusion - 1) * mean_le < max_2r:
      warnings.warn(
        f"SelfContact: nearest included pair gap {(adjacency_exclusion - 1) * mean_le:.2E} "
        f"< contact threshold 2r = {max_2r:.2E}. Spurious forces will fire on the ref. "
        f"config. Try adjacency_exclusion >= {int(np.ceil(max_2r / mean_le)) + 1}.",
        stacklevel=2,
      )

    pairs = np.array(
      [
        (aa, bb)
        for aa in range(num_elems - adjacency_exclusion)
        for bb in range(aa + adjacency_exclusion, num_elems)
      ],
      dtype=np.int32,
    )  # (v, p)
    trans_idx = np.concatenate([np.arange(n_t), np.arange(n_dn, n_dn + n_t)])

    pair_trans_dof_mat = np.stack(
      [
        rod.elem_dof_mat[pairs[:, 0]][:, trans_idx],
        rod.elem_dof_mat[pairs[:, 1]][:, trans_idx],
      ],
      axis=1,
    )  # (num_valid_pairs, 2, 2*n_t)

    elem_trans_dof_mat = rod.elem_dof_mat[:, trans_idx]  # (e, 2*n_t)
    sc_trans_iK = np.repeat(elem_trans_dof_mat, repeats=2 * n_t, axis=1).ravel()
    sc_trans_jK = np.tile(elem_trans_dof_mat, reps=(1, 2 * n_t)).ravel()

    # A broad phase that keeps every pair only adds the cost of the check.
    if max_candidate_pairs is not None and max_candidate_pairs >= pairs.shape[0]:
      max_candidate_pairs = None

    return cls(
      pairs=pairs,
      pair_trans_dof_mat=pair_trans_dof_mat,
      num_dofs_per_element=rod.num_dofs_per_element,
      num_dofs_per_node=rod.num_dofs_per_node,
      num_translational_dofs_per_node=rod.num_translational_dofs_per_node,
      sc_trans_iK=sc_trans_iK,
      sc_trans_jK=sc_trans_jK,
      stiffness=stiffness,
      smoothing_fraction=smoothing_fraction,
      max_candidate_pairs=max_candidate_pairs,
    )


def compute_self_contact_stiffness(
  rod: _mesher.Rod,
  youngs_modulus: float,
  stiffness_ratio: float = 0.5,
) -> float:
  """Physically-scaled self-contact penalty modulus k_sc [Pa].

  Sets the contact spring k_sc·L_e to a fixed fraction of the element bending
  stiffness E·I/L_e³, so contact is stiff enough to resist penetration yet stays
  below the structural stiffness (keeps the lumped-tangent valid and conditioning
  bounded). Concretely:

      k_sc · L_e = stiffness_ratio · (E · I_min / L_e³)
      =>  k_sc   = stiffness_ratio · E · I_min / L_e⁴

  Args:
    rod: `Rod` providing `elem_moi_tensor` and `elem_arc_lengths`.
    youngs_modulus: Representative Young's modulus E [Pa]. For an incompressible
      neo-Hookean material E = 3G; pass 3× the stiffer phase's shear modulus.
    stiffness_ratio: β ∈ (0, 1). Fraction of bending stiffness. Higher → less
      penetration but worse conditioning; lower → gentler tangent. Default 0.5.

  Returns:
    k_sc [Pa], the `stiffness` attribute for `SelfContact`.
  """
  bending_moi = jnp.stack(
    [rod.elem_moi_tensor[:, 0, 0], rod.elem_moi_tensor[:, 1, 1]], axis=-1
  )
  moi_min = jnp.min(bending_moi)  # softest bending axis
  mean_elem_arc_len = jnp.mean(rod.elem_arc_lengths)
  return float(stiffness_ratio * youngs_modulus * moi_min / mean_elem_arc_len**4)


def _pair_self_contact_energy(
  u_trans_pair: jnp.ndarray,
  pair_ref_node_coords: jnp.ndarray,
  geom_mean_arc_length: jnp.ndarray,
  radius_sum: jnp.ndarray,
  self_contact: "SelfContact",
) -> jnp.ndarray:
  """Self-contact penalty potential energy for one element pair [N·m].

  The energy is:

                  U_{ab} = (k_sc / 2) · ρ(ε, w)² · L_{ab}

  where:

    ε      = (r_a + r_b) − d_min     overlap depth [m]
    ρ(ε,w) = w · softplus(ε / w)     smooth penetration [m]; → max(ε, 0) as w → 0
    L_{ab} = sqrt(L_a · L_b)         geometric mean element arc-length [m]
    w      = smoothing_fraction · (r_a + r_b)   softplus transition width [m]

  Args:
    u_trans_pair: Concatenated incremental translational dofs for the two elems, shape
      `(2 · num_trans_dofs_per_elem,)` [m]. First `num_trans_dofs_per_elem` entries are
      element a's dofs; remaining entries are element b's.
    pair_ref_node_coords: Reference node positions for both elements, of shape
      `(num_elems_per_pair, num_nodes_per_elem=2, num_translational_dofs_per_node)` [m].
    geom_mean_arc_length: Geometric mean of the two element arc-lengths, sqrt(L_a · L_b)
      of shape `()` [m]. Scales the energy with a representative contact length.
    radius_sum: Sum of contact radii r_a + r_b, shape `()` [m]. Sets the threshold at
      which the penalty activates.
    self_contact: `SelfContact` instance.

  Returns:
    Scalar contact potential U_ab `()` [N·m].
  """
  # num_(d)ofs_per_elem
  n_t = self_contact.num_translational_dofs_per_node

  # Deformed endpoints in ℝ^{n_trans}: reference position plus incremental translation.
  p0 = pair_ref_node_coords[0, 0, :] + u_trans_pair[0 * n_t : 1 * n_t]
  p1 = pair_ref_node_coords[0, 1, :] + u_trans_pair[1 * n_t : 2 * n_t]
  q0 = pair_ref_node_coords[1, 0, :] + u_trans_pair[2 * n_t : 3 * n_t]
  q1 = pair_ref_node_coords[1, 1, :] + u_trans_pair[3 * n_t : 4 * n_t]

  min_separation = self_contact.min_separation_fraction * radius_sum
  d_min = jnp.maximum(
    _segment_segment_distance(p0=p0, p1=p1, q0=q0, q1=q1), min_separation
  )

  smoothing_width = self_contact.smoothing_fraction * radius_sum
  overlap = radius_sum - d_min
  smooth_pen = jnp.where(
    overlap <= 0.0,
    0.0,
    jnp.where(
      overlap < smoothing_width,
      overlap**3 / smoothing_width**2 - overlap**4 / (2.0 * smoothing_width**3),
      overlap - 0.5 * smoothing_width,
    ),
  )

  return 0.5 * self_contact.stiffness * jnp.square(smooth_pen) * geom_mean_arc_length


def _broad_phase_pair_gaps(
  u_elems: jnp.ndarray,
  rod: _mesher.Rod,
  self_contact: SelfContact,
) -> jnp.ndarray:
  """Cheap lower bound on the surface gap of every valid pair [m].

  Each deformed element is enclosed in a ball centred at its midpoint m with radius
  half its deformed length h. Every point of the segment lies inside its ball, so

                  d_min(a, b) ≥ ‖m_a − m_b‖ − h_a − h_b

  and the returned gap ‖m_a − m_b‖ − h_a − h_b − (r_a + r_b) is ≤ d_min − (r_a + r_b).
  A pair with a positive gap has zero overlap, so its energy, force and tangent are
  exactly zero. The gaps only select pairs and are not differentiated.

  Args:
    u_elems: Incremental dof array, of shape `(num_elems, num_dofs_per_element)`.
    rod: `Rod` instance.
    self_contact: `SelfContact` instance for the rod.

  Returns:
    Gap lower bounds, shape `(num_valid_pairs,)` [m].
  """
  n_t = self_contact.num_translational_dofs_per_node
  n_d = self_contact.num_dofs_per_node
  trans_idx = np.concatenate([np.arange(n_t), np.arange(n_d, n_d + n_t)])

  u_trans = jax.lax.stop_gradient(u_elems)[:, trans_idx].reshape(-1, 2, n_t)
  node_coords = rod.elem_node_coords[:, :, :n_t] + u_trans  # (e, 2, n_t)
  midpoints = 0.5 * (node_coords[:, 0] + node_coords[:, 1])
  half_lengths = 0.5 * jnp.linalg.norm(node_coords[:, 1] - node_coords[:, 0], axis=-1)

  aa, bb = self_contact.pairs[:, 0], self_contact.pairs[:, 1]
  center_dists = jnp.linalg.norm(midpoints[aa] - midpoints[bb], axis=-1)
  radius_sums = rod.contact_radii[aa] + rod.contact_radii[bb]
  return center_dists - half_lengths[aa] - half_lengths[bb] - radius_sums


def _warn_on_candidate_overflow(
  num_candidates: np.ndarray,
  max_candidate_pairs: int,
) -> None:
  """Host callback warning when the broad phase drops pairs that may be in contact."""
  # Batched under vmap, so report the worst case across the batch.
  num_candidates = int(np.max(num_candidates))
  if num_candidates > max_candidate_pairs:
    warnings.warn(
      f"SelfContact: {num_candidates} pairs may be in contact but "
      f"max_candidate_pairs = {max_candidate_pairs}. Pairs beyond the closest "
      f"{max_candidate_pairs} are dropped and their contact forces are missing. "
      f"Increase max_candidate_pairs."
    )


def compute_self_contact_forces_and_tangent(
  u_elems: jnp.ndarray,
  rod: _mesher.Rod,
  self_contact: SelfContact,
  compute_jacobian: bool,
) -> tuple[jnp.ndarray, jax_sprs.BCOO | None]:
  """Self-contact force vector and lumped diagonal tangent for all valid pairs.

  For each non-adjacent element pair (a, b) evaluates the contact potential U_{ab}
  and computes:

    Forces  : first derivative  ∂U_{ab}/∂u_pair  scattered into global DOF vector.
    Tangent : second derivative ∂²U_{ab}/∂u_pair² diagonal blocks scattered into
              global sparse matrix.

  The consistent contact tangent for pair (a, b) is the full (2nₑ × 2nₑ) Hessian:

    ∇²_{(u_a, u_b)} U_{ab} = [[H_aa,  H_ab],
                               [H_ba,  H_bb]]

  Assembling the cross-blocks H_ab and H_ba would couple every valid element pair
  in the global matrix, expanding the sparsity pattern from banded O(N) to dense
  O(N²). Retaining only the diagonal blocks H_aa and H_bb (lumped contact tangent)
  keeps K_total = K_elastic + K_contact with the identical sparsity pattern as
  K_elastic. The cross-block error is O(k_sc · L / K_elastic) per Newton-Raphson step
  and is corrected by subsequent iterations. Furthermore, the contact force is impacted
  only by the translational dofs and not the rotational dofs. We therefore extract only
  the translational dofs for the contact force and tangent calculations. This further
  reduces the computational cost and complexity of the contact model.

  When `self_contact.max_candidate_pairs` is set, a broad phase first keeps only the K
  pairs closest to contact (see `_broad_phase_pair_gaps`) and the exact evaluation runs
  on those alone.

  Args:
    u_elems: Incremental translational and rotational DOF array, of shape
      `(num_elems, num_dofs_per_element)` [m, rad]. Only the translational dofs are used
      in the contact force and tangent calculations.
    rod: `Rod` instance for which we are computing the self-contact.
    self_contact: `SelfContact` instance for the rod.
    compute_jacobian: When `True`, assembles and returns the lumped diagonal
      contact tangent in BCOO format. When `False`, returns `None`.

  Returns:
    Tuple `(f_contact, k_contact)`:
      - `f_contact`: Global self-contact force vector `(num_dofs,)` [N, N·m].
      - `k_contact`: Lumped diagonal contact tangent, BCOO shape
        `(num_dofs, num_dofs)`. `None` when `compute_jacobian=False`.
  """
  # num_(p)airs,
  pairs = self_contact.pairs  # (p, 2)
  pair_trans_dof_mat = self_contact.pair_trans_dof_mat  # (p, 2, 2*n_t)
  n_t = self_contact.num_translational_dofs_per_node
  n_d = self_contact.num_dofs_per_node

  max_candidates = self_contact.max_candidate_pairs
  if max_candidates is not None:
    pair_gaps = _broad_phase_pair_gaps(u_elems, rod, self_contact)
    jax.debug.callback(
      ft.partial(_warn_on_candidate_overflow, max_candidate_pairs=max_candidates),
      jnp.sum(pair_gaps <= 0.0),
    )
    _, candidate_idx = jax.lax.top_k(-pair_gaps, max_candidates)  # K smallest gaps
    pairs = jnp.asarray(pairs)[candidate_idx]  # (K, 2)
    pair_trans_dof_mat = jnp.asarray(pair_trans_dof_mat)[candidate_idx]

  trans_idx = np.concatenate([np.arange(n_t), np.arange(n_d, n_d + n_t)])

  u_trans_pairs = jnp.concatenate(
    [u_elems[pairs[:, 0]][:, trans_idx], u_elems[pairs[:, 1]][:, trans_idx]],
    axis=-1,
  )  # (p, 4*n_t)

  pair_ref_node_coords = rod.elem_node_coords[pairs, :, :n_t]

  geom_mean_arc_lengths = jnp.sqrt(
    rod.elem_arc_lengths[pairs[:, 0]] * rod.elem_arc_lengths[pairs[:, 1]]
  )  # (p,)

  radius_sums = rod.contact_radii[pairs[:, 0]] + rod.contact_radii[pairs[:, 1]]  # (p,)

  in_axes = (0, 0, 0, 0, None)

  vmap_args = (
    u_trans_pairs,
    pair_ref_node_coords,
    geom_mean_arc_lengths,
    radius_sums,
    self_contact,
  )

  pair_forces = jax.vmap(
    jax.grad(_pair_self_contact_energy, argnums=0),
    in_axes=in_axes,
  )(*vmap_args)

  f_contact = (
    jnp.zeros(rod.num_dofs)
    .at[pair_trans_dof_mat[:, 0, :]]
    .add(pair_forces[:, : 2 * n_t])
    .at[pair_trans_dof_mat[:, 1, :]]
    .add(pair_forces[:, 2 * n_t :])
  )

  if not compute_jacobian:
    return f_contact, None

  pair_hessians = jax.vmap(
    jax.hessian(_pair_self_contact_energy, argnums=0),
    in_axes=in_axes,
  )(*vmap_args)

  h_aa = -pair_hessians[:, : 2 * n_t, : 2 * n_t]
  h_bb = -pair_hessians[:, 2 * n_t :, 2 * n_t :]

  elem_h_aa = jax.ops.segment_sum(h_aa, pairs[:, 0], num_segments=rod.num_elems)
  elem_h_bb = jax.ops.segment_sum(h_bb, pairs[:, 1], num_segments=rod.num_elems)
  k_data = (elem_h_aa + elem_h_bb).reshape(-1)  # (e * (2*n_t)²,)

  sc_idx = (
    np.stack([self_contact.sc_trans_iK, self_contact.sc_trans_jK], axis=0)
    .astype(np.int32)
    .T
  )
  return f_contact, jax_sprs.BCOO(
    (k_data, sc_idx),
    shape=(rod.num_dofs, rod.num_dofs),
  )


@dataclasses.dataclass
class ObstacleMovement:
  """Prescribed rigid-body poses of an obstacle at each load step.

  Attributes:
    translations: `(num_steps, num_dim)` [m]. Obstacle origin in the lab frame.
    rotation_matrices: `(num_steps, num_dim, num_dim)`. Body-to-lab rotation at each
      step. Each entry must be orthonormal with det = +1; build them with
      `_utils.compute_rodrigues_matrix`.
  """

  translations: jnp.ndarray
  rotation_matrices: jnp.ndarray

  def __post_init__(self):
    num_steps, num_dim = self.translations.shape
    expected_shape = (num_steps, num_dim, num_dim)
    if self.rotation_matrices.shape != expected_shape:
      raise ValueError(
        f"rotation_matrices must have shape {expected_shape} to match "
        f"translations of shape {self.translations.shape}, got "
        f"{self.rotation_matrices.shape}."
      )

  @property
  def num_steps(self) -> int:
    """Returns the number of load steps covered by this schedule."""
    return self.translations.shape[0]

  @classmethod
  def from_translations(cls, translations: jnp.ndarray) -> "ObstacleMovement":
    """Builds a translation-only schedule with identity rotations at every step.

    Args:
      translations: `(num_steps, num_dim)` [m] obstacle origin at each load step.

    Returns:
      `ObstacleMovement` with identity rotations.
    """
    num_steps, num_dim = translations.shape
    return cls(
      translations=translations,
      rotation_matrices=jnp.broadcast_to(
        jnp.eye(num_dim), (num_steps, num_dim, num_dim)
      ),
    )


@dataclasses.dataclass
class Obstacle:
  """Obstacle geometry encoded as a background signed distance field grid.

  Maps a structured spatial grid of signed distance values to a continuous,
  differentiable field via cubic spline interpolation. Obstacle surfaces are
  the zero level set of φ. The sign convention is φ > 0 in free space and
  φ < 0 inside obstacles.

  `compute_sdf` and `compute_sdf_gradient` batch queries over arbitrary
  arrays of continuous positions. Gradients are exact (autodiff through the
  cubic spline), so no finite-difference stencil is needed for contact normals.

  Attributes:
    sdf_grid: Discrete signed distance values, shape `(Nx, Ny)` in 2D or
      `(Nx, Ny, Nz)` in 3D [m]. Must follow the module-level φ convention.
    physical_origin: Global Cartesian coordinate of grid node `[0, 0, ...]`,
      shape `(num_dim,)` [m]. Typically the bottom-left-front corner.
    spacing: Uniform physical step between adjacent grid nodes along each axis, shape
      `(num_dim,)` [m].
    wall_penalty_stiffness: Stiffness parameter for the penalty force applied when the
      rod comes into contact with the environment.
    extrap_fill_value: Signed distance value [m] returned by `sdf_at_point` for query
      points that fall outside the grid bounding box. Defaults to 1, indicating that
      the point is in free space. Use a negative value to indicate that the point is
      inside an obstacle (in contact).
    movement: Optional `ObstacleMovement` instance describing the rigid-body
      translation and rotation of the obstacle at each load step. If `None`, the
      obstacle is fixed in the lab frame.
  """

  sdf_grid: jnp.ndarray
  physical_origin: jnp.ndarray
  spacing: jnp.ndarray
  wall_penalty_stiffness: float
  extrap_fill_value: float = 1.0
  movement: Optional[ObstacleMovement] = None

  def sdf_at_point(self, point: jnp.ndarray) -> jnp.ndarray:
    """Evaluates the cubic spline SDF interpolant at a single query point.

    Dispatches to `interpax.interp2d` or `interpax.interp3d` based on the dimensionality
    of `point`. Extrapolation outside the grid is enabled and set to the value of
    `extrap_fill_value` (by default 1.0, indicating free space).

    Args:
      point: Query coordinate, shape `(num_dim,)` [m]. Requires `num_dim ∈ {2, 3}`.

    Returns:
      Scalar signed distance φ(point) [m].
    """
    grid_shape = self.sdf_grid.shape

    if self.sdf_grid.ndim == 2:
      x_knots = self.physical_origin[0] + jnp.arange(grid_shape[0]) * self.spacing[0]
      y_knots = self.physical_origin[1] + jnp.arange(grid_shape[1]) * self.spacing[1]
      val = interpax.interp2d(
        point[0:1],
        point[1:2],
        x_knots,
        y_knots,
        self.sdf_grid,
        method="cubic",
        extrap=self.extrap_fill_value,
      )
      return val[0]

    elif self.sdf_grid.ndim == 3:
      x_knots = self.physical_origin[0] + jnp.arange(grid_shape[0]) * self.spacing[0]
      y_knots = self.physical_origin[1] + jnp.arange(grid_shape[1]) * self.spacing[1]
      z_knots = self.physical_origin[2] + jnp.arange(grid_shape[2]) * self.spacing[2]
      val = interpax.interp3d(
        point[0:1],
        point[1:2],
        point[2:3],
        x_knots,
        y_knots,
        z_knots,
        self.sdf_grid,
        method="cubic",
        extrap=self.extrap_fill_value,
      )
      return val[0]

    else:
      raise ValueError("Only 2D and 3D configuration fields are supported.")

  @classmethod
  def from_image(
    cls,
    image: np.ndarray,
    physical_origin: jnp.ndarray,
    spacing: jnp.ndarray,
    img_type: Literal["sdf", "binary"],
    wall_penalty_stiffness: float,
    extrap_fill_value: float = 1.0,
    movement: Optional[ObstacleMovement] = None,
  ) -> "Obstacle":
    """Constructs an `Obstacle` from a spatial domain array.

    Args:
      image: Spatial domain array of shape `(Nx, Ny)` or `(Nx, Ny, Nz)`.
        For `img_type="binary"`: values must be strictly 0 (free space) or
        1 (solid obstacle). For `img_type="sdf"`: precomputed signed distance
        values following the module-level φ convention (φ > 0 in free space,
        φ < 0 inside obstacles).
      physical_origin: Coordinate of grid node `[0, 0, ...]`, shape `(num_dim,)` [m].
      spacing: Physical voxel or pixel size along each axis, shape `(num_dim,)` [m].
      img_type: `"binary"` to compute SDF via `skfmm.distance`, or `"sdf"` to use
        precomputed values directly.
      wall_penalty_stiffness: Stiffness parameter for the penalty force applied when the
        rod comes into contact with the environment.
      extrap_fill_value: Signed distance returned for query points outside the grid
        bounding box. Default 1.0 m (free space).
      movement: Optional `ObstacleMovement` instance describing the rigid-body
        translation and rotation of the obstacle at each load step. If `None`, the
        obstacle is fixed in the lab frame.

    Returns:
      Initialized `Obstacle` instance.
    """
    if img_type == "binary":
      # Free space (image = 0) receives phi = +0.5; obstacles (image > 0)
      # receive phi = -0.5.
      phi = np.where(image > 0, -0.5, 0.5)
      grid = jnp.asarray(skfmm.distance(phi, dx=np.asarray(spacing, dtype=np.float64)))
    elif img_type == "sdf":
      grid = jnp.asarray(image)
    else:
      raise ValueError("img_type must be 'sdf' or 'binary'.")
    return cls(
      sdf_grid=grid,
      physical_origin=physical_origin,
      spacing=spacing,
      wall_penalty_stiffness=wall_penalty_stiffness,
      extrap_fill_value=extrap_fill_value,
      movement=movement,
    )

  @classmethod
  def from_mesh(
    cls,
    mesh: trimesh.Trimesh,
    grid_shape: tuple[int, int, int],
    wall_penalty_stiffness: float,
    padding: float,
    extrap_fill_value: float = 1.0,
    movement: Optional[ObstacleMovement] = None,
  ):
    """Constructs an `Obstacle` from a trimesh.

    NOTE: Be careful whether what the mesh represents to ensure the correct sign of the
    SDF values. The sign should be positive in free space and negative inside obstacles.

    Args:
      mesh: A trimesh object consisting of a surface mesh. Note that with multiple
        objects, the trimesh should be forced to a mesh.
      grid_shape: Number of voxels along (x, y, z), i.e. `(nelx, nely,
        nelz)`.
      wall_penalty_stiffness: Stiffness parameter for the penalty force applied when the
        rod comes into contact with the environment.
      padding: Isotropic distance [m] by which the mesh bounding box is expanded
        before sampling. Must be at least the largest element contact radius of any
        rod that will interact with this obstacle, and two to three times that is
        recommended so the activation band is fully resolved.
      extrap_fill_value: Signed distance returned for query points outside the grid
        bounding box. Must exceed the largest contact radius so that leaving the grid
        reads as free space.Default 1.0 m (free space).
      movement: Optional `ObstacleMovement` instance describing the rigid-body
        translation and rotation of the obstacle at each load step. If `None`, the
        obstacle is fixed in the lab frame.

    Returns:
      Initialized `Obstacle` instance.
    """
    bbox_min = np.asarray(mesh.bounds[0], dtype=np.float64) - padding
    bbox_max = np.asarray(mesh.bounds[1], dtype=np.float64) + padding
    spacing = (bbox_max - bbox_min) / np.asarray(grid_shape, dtype=np.float64)

    axes = [
      bbox_min[i] + (np.arange(grid_shape[i]) + 0.5) * spacing[i] for i in range(3)
    ]
    grid_x, grid_y, grid_z = np.meshgrid(*axes, indexing="ij")

    query_points = np.stack(
      [grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1
    )  # (nelx*nely*nelz, 3)
    sdf_fn = pysdf.SDF(
      np.asarray(mesh.vertices, dtype=np.float32),
      np.asarray(mesh.faces, dtype=np.int32),
    )
    # pysdf is positive inside the mesh; this module is positive in free space.
    sdf_flat = -np.asarray(sdf_fn(query_points), dtype=np.float64)

    return cls(
      sdf_grid=jnp.asarray(sdf_flat.reshape(grid_shape)),
      physical_origin=jnp.asarray(bbox_min + 0.5 * spacing),
      spacing=jnp.asarray(spacing),
      wall_penalty_stiffness=wall_penalty_stiffness,
      extrap_fill_value=extrap_fill_value,
      movement=movement,
    )

  def compute_sdf(
    self,
    points: jnp.ndarray,
    obj_trans: jnp.ndarray,
    obj_rot_mat: jnp.ndarray,
  ) -> jnp.ndarray:
    """Evaluates the interpolated signed distance at continuous coordinates.

    Query points arrive in the lab frame; the signed distance grid is defined in the
    obstacle body frame. The obstacle pose maps body to lab as x_lab = R x_body + T,
    so the inverse map x_body = Rᵀ (x_lab − T) is applied here.

    Args:
      points: Lab-frame query coordinates, shape `(num_pts, num_dim)` [m].
      obj_trans: Obstacle translation T, shape `(num_dim,)` [m].
      obj_rot_mat: Obstacle rotation R ∈ SO(d), shape `(num_dim, num_dim)`.

    Returns:
      Signed distance values φ(x), shape `(num_pts,)` [m]. Positive in free space,
        negative inside obstacles.
    """
    points_query = _utils.affine_transform(
      points, obj_trans[None, :], obj_rot_mat, "GLOBAL_TO_LOCAL"
    )
    return jax.vmap(self.sdf_at_point, in_axes=(0,))(points_query)

  def compute_sdf_gradient(
    self,
    points: jnp.ndarray,
    obj_trans: jnp.ndarray,
    obj_rot_mat: jnp.ndarray,
  ) -> jnp.ndarray:
    """Computes ∇φ at continuous coordinates via autodiff through the spline.

    Since φ > 0 in free space, ∇φ points from the wall interior toward free
    space — directly giving the contact repulsion direction.

    Query points arrive in the lab frame; the signed distance grid is defined in the
    obstacle body frame. The obstacle pose maps body to lab as x_lab = R x_body + T,
    so the inverse map x_body = Rᵀ (x_lab − T) is applied here.

    Args:
      points: Lab-frame query coordinates, shape `(num_pts, num_dim)` [m].
      obj_trans: Obstacle translation T, shape `(num_dim,)` [m].
      obj_rot_mat: Obstacle rotation R ∈ SO(d), shape `(num_dim, num_dim)`.

    Returns:
      Gradient vectors ∇φ(x), shape `(num_pts, num_dim)` [m⁻¹].
    """
    points_query = _utils.affine_transform(
      points, obj_trans[None, :], obj_rot_mat, "GLOBAL_TO_LOCAL"
    )
    grad_fn = jax.vmap(jax.grad(self.sdf_at_point, argnums=0), in_axes=(0,))
    return jnp.einsum("ij,nj->ni", obj_rot_mat, grad_fn(points_query))

  def pose_at_step(self, step_idx: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the obstacle translation and rotation at one load step.

    Falls back to the identity pose when no movement is attached, so a static
    obstacle reduces exactly to the untransformed case.

    Args:
      step_idx: Scalar integer array, the current load step.

    Returns:
      `(obj_trans, obj_rot_mat)` of shapes `(num_dim,)` [m] and
        `(num_dim, num_dim)`.
    """
    num_dim = self.sdf_grid.ndim
    if self.movement is None:
      return jnp.zeros(num_dim), jnp.eye(num_dim)
    return (
      self.movement.translations[step_idx],
      self.movement.rotation_matrices[step_idx],
    )


def _compute_elem_wall_contact_energy(
  u_e: jnp.ndarray,
  ref_center: jnp.ndarray,
  elem_length: jnp.ndarray,
  elem_contact_radius: jnp.ndarray,
  heaviside_smooth_width: jnp.ndarray,
  obstacle: Obstacle,
  num_nodes_per_element: int,
  obj_trans: jnp.ndarray,
  obj_rot_mat: jnp.ndarray,
) -> jnp.ndarray:
  """Wall contact potential energy for one element [N·m].

  Implements the conservative potential from Gazzola (2018) Eq. 4.8:

    Full Gazzola equation (quasistatic limit only currently implemented):

      f^w = H_σ(ε) · (−F_⊥  +  k_w · ε  −  γ_w · v_⊥) · n̂   [N/m]

    where:
      f^w  — contact force line density [N/m] at the element centre.
      H_σ  — smooth Heaviside of width σ [m]: H_σ(ε) → 0 for ε ≪ 0
               (no contact), H_σ(ε) → 1 for ε ≫ 0 (full contact).
      ε    — overlap depth [m]: ε = r_e − φ(x_c), positive when the rod
               surface has penetrated the wall.
      r_e  — rod cross-section contact radius [m].
      φ    — signed distance field value at x_c; φ > 0 in free space.
      x_c  — deformed element centre position [m].
      F_⊥  — component of non-contact external force normal to the wall
               [N/m]; used by explicit integrators; zero for implicit NR.
      k_w  — wall penalty stiffness [N/m²].
      γ_w  — wall normal damping coefficient [N·s/m²].
      v_⊥  — normal component of element velocity [m/s], positive toward
               the wall: v_⊥ = v_elem · n̂.
      n̂    — unit outward surface normal: n̂ = ∇φ / |∇φ| [dimensionless].
      L_e  — element arc-length [m].  Nodal force = (L_e / 2) · f^w
               (midpoint quadrature, two bounding nodes).

    Currently implemented (F_⊥ = 0, γ_w = 0):

                    U_e = (k_w / 2) · ρ(ε, σ)² · L_e

    where ρ(ε, σ) = σ · softplus(ε / σ) [m]. ρ is the scaled softplus that approaches
    max(ε, 0) for σ → 0 and is strictly ≥ 0 (no attractive force in free space). The
    force recovers:

            −∂U_e/∂u_e  ≈  k_w · max(ε, 0) · n̂ · L_e   for σ ≪ r_e

  Args:
    u_e: Element DOF vector `(num_nodes_per_element · num_dofs_per_node,)`
      [m, rad]. The translation dofs of node 1 are the first `num_spatial_dims` indices
      in the DOF vector, followed by the rotational dofs of node 1, then the translation
      dofs of node 2, and so on.
    ref_center: Element centre in the reference configuration of shape `(3,)` [m].
    elem_length: Element arc-length L_e `()` [m].
    elem_contact_radius: Cross-section contact radius r_e `()` [m].
    heaviside_smooth_width: Softplus width σ_e `()` [m].
    obstacle: `Obstacle` data structure.
    num_nodes_per_element: `rod.num_nodes_per_element` (always 2 for linear
      Cosserat rod elements).
    obj_trans: Total rod translation T, of shape `(num_dim,)`.
    obj_rot_mat: Total rod rotation R ∈ SO(d), of shape `(num_dim, num_dim)`.

  Returns:
    Scalar contact potential U_e [N·m].
  """

  num_dofs_per_node = u_e.shape[0] // num_nodes_per_element
  # Spatial dimensions: 2 for 2D obstacle, 3 for 3D (static at trace time).
  num_spatial_dims = obstacle.sdf_grid.ndim

  x_c_deformed = ref_center[:num_spatial_dims] + 0.5 * (
    u_e[:num_spatial_dims]
    + u_e[num_dofs_per_node : num_dofs_per_node + num_spatial_dims]
  )  # (num_spatial_dims,)

  x_c_query = _utils.affine_transform(
    x_c_deformed, obj_trans, obj_rot_mat, "GLOBAL_TO_LOCAL"
  )
  signed_dist = obstacle.sdf_at_point(x_c_query)
  overlap_depth = elem_contact_radius - signed_dist  # ε = r_e − φ(x_c)
  smooth_penetration = jnp.where(
    overlap_depth <= 0.0,
    0.0,
    jnp.where(
      overlap_depth < heaviside_smooth_width,
      overlap_depth**3 / heaviside_smooth_width**2
      - overlap_depth**4 / (2.0 * heaviside_smooth_width**3),
      overlap_depth - 0.5 * heaviside_smooth_width,
    ),
  )

  return (
    0.5 * obstacle.wall_penalty_stiffness * jnp.square(smooth_penetration) * elem_length
  )


def compute_wall_contact_forces_and_tangent(
  u_elems: jnp.ndarray,
  rod: _mesher.Rod,
  obstacle: Obstacle,
  compute_jacobian: bool,
  heaviside_smooth_width_rel: float = 0.25,
  obj_trans: Optional[jnp.ndarray] = None,
  obj_rot_mat: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
  """Compute wall contact force vector and K_contact for all elements.

  Returns the contact force vector and the contact Jacobian (if requested) for the
  rod contacting with a wall or other obstacles.

  Smooth Heaviside width is specified non-dimensionally:

             σ_e = σ_rel · r_e   [m]

  A value of 0.05 activates contact over a 5 % penetration band around
  ε = 0, independent of the rod's physical length scale.

  Args:
    u_elems: Element DOF array `(num_elems, num_dofs_per_elem)` [m, rad].
    rod: `Rod` instance.  Requires `.elem_centers`, `.elem_lengths`,
      `.contact_radii`, `.elem_dof_mat`, `.iK`, `.jK`, `.num_dofs`,
      `.num_nodes_per_element`.
    obstacle: `Obstacle` data structure.
    compute_jacobian: Return K_contact only when `True`.
    heaviside_smooth_width_rel: Non-dimensional width σ_rel = σ / r.
      Dimensionalised per-element as σ_e = σ_rel · r_e [m].
    obj_trans: Total rod translation T, of shape `(num_dim,)`. A default value of None
      indicates that the rod is not translated.
    obj_rot_mat: Total rod rotation R ∈ SO(d), of shape `(num_dim, num_dim)`. A default
      value of None indicates that the rod is not rotated.

  Returns:
    Tuple `(f_contact, k_contact)` where `k_contact` is `None` when
    `compute_jacobian` is `False`.
  """
  # (e)lement, (o)=dof_per_elem
  smooth_widths = heaviside_smooth_width_rel * rod.contact_radii  # (e,)
  num_dim = obstacle.sdf_grid.ndim

  if obj_trans is None:
    obj_trans = jnp.zeros(num_dim)
  if obj_rot_mat is None:
    obj_rot_mat = jnp.eye(num_dim)

  _f_int_fn = jax.grad(_compute_elem_wall_contact_energy, argnums=0)

  vmap_args = (
    u_elems,
    rod.elem_centers,
    rod.elem_arc_lengths,
    rod.contact_radii,
    smooth_widths,
    obstacle,
    rod.num_nodes_per_element,
    obj_trans,
    obj_rot_mat,
  )
  vmap_ax = (0, 0, 0, 0, 0, None, None, None, None)

  elem_force = jax.vmap(_f_int_fn, in_axes=vmap_ax)(*vmap_args)  # (e, o)
  f_contact = jnp.zeros(rod.num_dofs).at[rod.elem_dof_mat].add(elem_force)

  if not compute_jacobian:
    return f_contact, None

  elem_jac = -jax.vmap(jax.jacfwd(_f_int_fn, argnums=0), in_axes=vmap_ax)(
    *vmap_args
  )  # (e, o, o)
  node_idx = np.stack((rod.iK, rod.jK)).astype(np.int32).T
  k_contact = jax_sprs.BCOO(
    (elem_jac.flatten(order="C"), node_idx),
    shape=(rod.num_dofs, rod.num_dofs),
  )
  return f_contact, k_contact


def compute_wall_forces_and_moments(
  rod: _mesher.Rod,
  obstacle: Obstacle,
  displacements: jnp.ndarray,
  step_idx: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Computes the nodal forces and moments the wall exerts on the rod.

  The load the wall exerts on the rod is split into the translational part (forces) and
  the rotational part (moments). 
  
  NOTE: The current formulation of external contact depends on element centres alone, 
  so the moments are zero. They are returned so that a surface-point or frictional 
  contact model needs no API change.

  Args:
    rod: `Rod` whose element dof map and contact radii define the wall energy.
    obstacle: `Obstacle` carrying the movement used in the solve.
    displacements: Array of shape `(num_dofs,)` or `(num_load_steps, num_dofs)` [m, rad]
      of converged states.
    step_idx: Load step whose obstacle pose is used for a one-dimensional `disp`,
      ignored for a history.

  Returns:
    A tuple `(forces, moments)`, each of shape `(num_nodes, 3)` [N] and [N m], with a
      leading `num_load_steps` axis when `displacements` is two-dimensional.
  """
  if displacements.ndim == 2:
    step_indices = jnp.arange(displacements.shape[0])
    single_state = ft.partial(compute_wall_forces_and_moments, rod, obstacle)
    return jax.vmap(single_state)(displacements, step_indices)

  obj_trans, obj_rot_mat = obstacle.pose_at_step(step_idx)
  elem_disp = displacements[rod.elem_dof_mat]
  internal, _ = compute_wall_contact_forces_and_tangent(
    u_elems=elem_disp,
    rod=rod,
    obstacle=obstacle,
    compute_jacobian=False,
    obj_trans=obj_trans,
    obj_rot_mat=obj_rot_mat,
  )
  nodal = internal.reshape(rod.num_nodes, rod.num_dofs_per_node)
  num_trans = rod.num_translational_dofs_per_node
  return -nodal[:, :num_trans], -nodal[:, num_trans:]
