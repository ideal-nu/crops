"""Utility functions."""

import enum
from typing import Literal, Sequence, Tuple, TypeAlias, Union

import chex
import jax
import jax.experimental.sparse as jax_sprs
import jax.numpy as jnp
import numpy as np
import shapely
from jax.typing import ArrayLike

GAS_CONSTANT = 8.31446261815324  # J/(mol·K)
PERMEABILITY_VACUUM = 4 * jnp.pi * 1e-7  # H/m (henries per meter)


class Director(enum.IntEnum):
  """Integer indices mapping material frame director axes.

      ^ d1 (cross-section transverse)             ▲ d₁
    |                                           |
    ============================                │
    |      length              |                │    ┌──------─┐
    ============================                │    │         │ <- rod cross-section
                            --> d3 (axial)      │    └──------─┘    at arc coordinate s
                                                │
                                                └──────────────► d₂.   (cross-section)

  Attributes:
    D1: First principal axis of the cross-section.
    D2: Second principal axis of the cross-section, orthogonal to d₁.
    D3: The third director axis, aligned with the tangent / axial direction of
      the rods centerline.
  """

  D1 = 0
  D2 = 1
  D3 = 2


class Translation(enum.Enum):
  """Translational degrees of freedom."""

  X = 0
  Y = 1
  Z = 2


class Rotation(enum.Enum):
  """Rotational degrees of freedom."""

  THETA_X = 0
  THETA_Y = 1
  THETA_Z = 2


Translate_or_Rotate: TypeAlias = Union[Translation, Rotation]


@chex.dataclass
class Extent:
  """Extent of a variable."""

  min: ArrayLike
  max: ArrayLike

  def __post_init__(self):
    if self.min > self.max:
      raise ValueError("min must be less than or equal to max")
    if self.range < 1e-12:
      print("Warning: The range of the extent is very small.")

  @property
  def range(self) -> ArrayLike:
    """Computes the total range (max - min) of the extent."""
    return self.max - self.min

  @property
  def center(self) -> ArrayLike:
    """Computes the exact midpoint or center of the defined extent."""
    return 0.5 * (self.min + self.max)

  def normalize_array(self, x: ArrayLike) -> ArrayLike:
    """Linearly normalize `x` using `extent` ranges."""
    return (x - self.min) / self.range

  def renormalize_array(self, x: ArrayLike) -> ArrayLike:
    """Recover array from linearly normalized `x` using `extent` ranges."""
    return x * self.range + self.min


class DesignLayout:
  """Maps one flat normalized design vector to named physical arrays and back.

  Each design variable is a block `(name, shape, extent)`. The flat vector holds the
  blocks flend to end in the given order, so block `b` occupies `[o_b, o_{b+1})` with
  `o_{b+1} = o_b + Π shape_b`. Every flat entry lies in `[0, 1]` and maps to its
  physical value through the block extent as `x_phys = x_min + x (x_max − x_min)`.

  Attributes:
    blocks: Tuple of `(name, shape, extent)` triples in flat-vector order. Names are
      unique, `shape` is a tuple of ints, and `extent` is an `Extent`.
    offsets: List of `num_blocks + 1` Python ints. Block `b` spans
      `[offsets[b], offsets[b + 1])` of the flat vector.
    num_design_var: Total length of the flat vector, equal to `offsets[-1]`.
  """

  def __init__(
    self,
    blocks: Sequence[tuple[str, tuple[int, ...], Extent]],
  ):
    """Stores the blocks and computes their offsets in the flat vector.

    Args:
      blocks: Sequence of `(name, shape, extent)` triples in the order they occupy the
        flat vector. Names must be unique.
    """
    self.blocks = tuple(blocks)
    sizes = [int(np.prod(shape)) for _, shape, _ in self.blocks]
    self.offsets = [0, *np.cumsum(sizes).tolist()]
    self.num_design_var = self.offsets[-1]

  def split(self, flat: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Cuts a flat vector into named blocks of their shapes without any scaling.

    A gradient with respect to the design vector shares its layout, so this also
    splits gradients, for example to report one norm per design variable.

    Args:
      flat: Array of shape `(num_design_var,)` of design variables or gradients.

    Returns:
      Dictionary mapping each block name to an array of that block's shape.
    """
    starts = self.offsets[:-1]
    stops = self.offsets[1:]
    return {
      name: flat[start:stop].reshape(shape)
      for (name, shape, _), start, stop in zip(self.blocks, starts, stops)
    }

  def to_physical(self, flat: jnp.ndarray) -> dict[str, jnp.ndarray]:
    """Splits a normalized design vector and maps every block onto its extent.

    Args:
      flat: Array of shape `(num_design_var,)` of design variables in `[0, 1]`.

    Returns:
      Dictionary mapping each block name to an array of that block's shape in
        physical units.
    """
    normalized = self.split(flat)
    return {
      name: extent.renormalize_array(normalized[name])
      for name, _, extent in self.blocks
    }

  def to_flat(self, physical: dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Normalizes named physical arrays and concatenates them in block order.

    Values outside an extent are not clipped, so an out of range start design stays
    visible rather than being silently moved.

    Args:
      physical: Dictionary mapping every block name to an array of that block's
        shape in physical units.

    Returns:
      Array of shape `(num_design_var,)` of normalized design variables.
    """
    parts = [
      extent.normalize_array(physical[name]).reshape(-1)
      for name, _, extent in self.blocks
    ]
    return jnp.concatenate(parts)


def safe_power(x: jax.Array, exp: float) -> jax.Array:
  """Compute the power `x**exp` with a safe check for negative/zero values.

  This function ensures that the input `x` is positive before applying the power
    operation. If `x` is negative or zero, it returns zero. This ensures that the
    power operation does not result in undefined behavior or complex numbers.

  Args:
    x: Input array.
    exp: Exponent value.

  Returns: The result of `x**exp` if `x` is positive, otherwise zero.
  """
  z = jnp.where(x <= 0.0, 1.0, x)
  result = jnp.where(x > 0.0, jnp.power(z, exp), 0.0)

  if exp == 0.0:  # If exp is exactly 0, override the result to 1.0
    return jnp.ones_like(x)
  return result


def safe_log(x: jax.Array) -> jax.Array:
  """Compute the natural logarithm of `x` with a safe check for non-positive values.

  Args:
    x: Input array.
  Returns: The natural logarithm of `x` if `x` is positive, otherwise zero.
  """
  z = jnp.where(x <= 0.0, 1.0, x)
  return jnp.where(x > 0.0, jnp.log(z), 0.0)


def safe_sqrt(x: jax.Array) -> jax.Array:
  """Compute the square root of x with a safe check for negative values.

  This function ensures that the input `x` is non-negative before applying the
    square root operation. If `x` is negative, it returns zero. This ensures that
    the square root operation does not result in undefined behavior.

  Args:
    x: Input array.

  Returns: The square root of `x` if `x` is non-negative, otherwise zero.
  """
  z = jnp.where(x <= 0.0, 1.0, x)
  return jnp.where(x > 0.0, jnp.sqrt(z), 0.0)


def safe_divide(x: jax.Array, y: jax.Array, eps: float = 1.0e-12) -> jax.Array:
  """Compute the division of x by y with a safe check for division by zero.

  This function ensures that the denominator `y` is non-zero before applying the
    division operation. If `y` is zero, it returns zero. This ensures that the
    division operation does not result in undefined behavior.

  Args:
    x: Numerator array.
    y: Denominator array.
    eps: Small value below which the absolute value of the denominator is
      treated as zero. Default is 1.0e-12.

  Returns: The result of `x / y` if `y` is non-zero, otherwise zero.
  """
  z = jnp.where(jnp.abs(y) < eps, 1.0, y)
  return jnp.where(jnp.abs(y) < eps, 0.0, x / z)


def safe_pnorm(x: jax.Array, p: float, axis: int):
  """Compute the p-norm of x with a safe check for negative values.

  This function ensures that the input `x` is non-negative before applying the
    p-norm operation. If `x` is negative, it returns zero. This ensures that the
  p-norm operation does not result in undefined behavior.

  The p-norm is defined as:
               ||x||_p = (sum_i(|x_i|^p))^(1/p)

    where |x_i| is the absolute value of the i-th element of x, and the sum is taken
    over the specified axis.

  The function is often used to compute a smooth approximation to the maximum (or min)
  of a set of values.

  Args:
    x: The input array.
    p: The Exponent value. The larger the value, the closer the p-norm is to the max.
      However, the problem becomes more nonlinear. A typical value is 6.0.
    axis: The axis along which the p-norm is computed.

  Returns: The p-norm of `x` computed in a safe manner along the specified axis.
  """
  sum_x = jnp.sum(safe_power(jnp.abs(x), p), axis=axis)
  return safe_power(sum_x, 1.0 / p)


def safe_normalize(
  arr: jnp.ndarray, axis: int = -1, eps: float = 1.0e-12, keepdims: bool = True
) -> jnp.ndarray:
  """Normalize `arr` along `axis`, guarding against zero-norm inputs.

  Divides `arr` by its L2 norm along `axis`. A `jax.lax.stop_gradient` wraps
  the denominator so that near-zero norms do not produce NaN in reverse-mode
  differentiation. The gradient flows through the numerator only, giving
  ∂(arr/‖arr‖)/∂arr ≈ 1/max(‖arr‖, ε) rather than the full unit-vector
  Jacobian. This approximation is finite everywhere.

  Args:
    arr: Input array of arbitrary shape. Shape is preserved in the output.
    axis: Axis along which to compute the L2 norm. Default -1 (last axis).
    eps: Small value below which the absolute value of the norm is treated as zero.
    keepdims: If `True`, retains reduced dimensions with size 1. If `False`, the
      reduced dimensions are removed from the output. Default is `True`.

  Returns:
    Array of the same shape as `arr` with unit norm along `axis` wherever
      ‖arr‖ ≥ ε. Returns `arr / ε`  (which scales down linearly to zero) for near-zero
      inputs.
  """
  norm = jnp.linalg.norm(arr, axis=axis, keepdims=keepdims)
  return arr / jax.lax.stop_gradient(jnp.maximum(norm, eps))


def inverse_sigmoid(y: jax.Array) -> jax.Array:
  """The inverse of the sigmoid function.

  The sigmoid function f:x->y is defined as:

           f(x) = 1 / (1 + exp(-x))

  The inverse sigmoid function g: y->x is defined as:

           g(y) = ln(y / (1 - y))

  For details see https://tinyurl.com/y7mr76hm

  Args:
    y: Assumed to be an array of output space values logically bounded in (0, 1).

  Returns:
    The pre-activation scalar or array corresponding to the input mapped backwards
      onto the entire real line.
  """
  return jnp.log(y / (1.0 - y))


def cosine_similarity(
  a: jax.typing.ArrayLike,
  b: jax.typing.ArrayLike,
  epsilon: jax.typing.ArrayLike = 1e-6,
  axis: Union[int, tuple[int, ...], None] = -1,
) -> jax.Array:
  """Computes the cosine similarity between targets and predictions.

  The cosine similarity is a measure of similarity between vectors defined
  as the cosine of the angle between them, which is also the inner product of
  those vectors normalized to have unit norm.

  Args:
    a: The primary vectors, with shape `[..., dim]`.
    b: The secondary vectors, with shape `[..., dim]`.
    epsilon: minimum value used to clip the squared norms in the denominator, for
      numerical stability. The squared norms are clipped to be at least ``epsilon``,
      so the effective minimum norm is ``sqrt(epsilon)``.
    axis: Axis or axes along which to compute.

  Returns:
    cosine similarity measures, with shape `[...]`. The values are in range
      [-1, 1].

  References:
    `Cosine similarity <https://en.wikipedia.org/wiki/Cosine_similarity>`_,
    Wikipedia.

  """

  a_norm = jnp.sqrt(jnp.square(a).sum(axis=axis, keepdims=True).clip(epsilon))
  b_norm = jnp.sqrt(jnp.square(b).sum(axis=axis, keepdims=True).clip(epsilon))

  a_unit = a / a_norm
  b_unit = b / b_norm
  return (a_unit * b_unit).sum(axis=axis)


def smooth_step(
  x: ArrayLike, center: ArrayLike, width: ArrayLike, eps: float = 1e-8
) -> jax.Array:
  """Computes a smooth step function, transitioning from 0 to 1.

  This function provides a smoothed, differentiable approximation of the Heaviside
    step function using the hyperbolic tangent. The transition is centered at
    `center` and has a characteristic width of `width`.

  Args:
    x: The input value(s) at which to evaluate the function.
    center: The midpoint of the transition.
    width: The width of the transition zone. The function moves from
      approximately 0 to 1 over an interval of a few multiples of `width`.
    eps: A small epsilon value to prevent division by zero for a zero width.

  Returns:
    The value of the smooth step function, ranging from 0 to 1.
  """
  x_center = x - center
  x_norm = x_center / (width + eps)
  return 0.5 * (1.0 + jnp.tanh(x_norm))


def _cdist(a: jax.Array, b: jax.Array) -> jax.Array:
  """Computes pairwise Euclidean distances between rows of a and b.

  Args:
    a: Array of shape (N, D) - N points, D dimensions.
    b: Array of shape (M, D) - M points, D dimensions.

  Returns:
    Array of shape (N, M) containing distances.
  """
  diff = a[:, jnp.newaxis, :] - b[jnp.newaxis, :, :]
  return jnp.linalg.norm(diff, axis=-1)


class Filters(enum.Enum):
  """Enum for different types of filters."""

  LINEAR = enum.auto()
  CIRCULAR = enum.auto()
  GAUSSIAN = enum.auto()


def create_density_filter(
  coords: jax.Array,
  cutoff_distance: float,
  filter_type: Filters = Filters.LINEAR,
  eps: float = 1e-12,
) -> jax_sprs.BCOO:
  """Creates a density filter to smoothen out the field.

  The density filter is to ensure that the obtained density fields do not have
  checkerboard patterns. This is common in density-based topology optimization problems.

  NOTE: For a rod we apply the filter only along the length. So this function expects
  we reshape say the pseudodensity so that the leading axis is num_elems (with the
  trailing axis being num_fibers). Filtering along in the cross section is not accounted
  for in this and if required, this function may need to be reconsidered.

  Args:
    coords: An array of shape (num_pts, num_dim) of the coordinates of the points.
    cutoff_distance: A float, the radius beyond which the filter has zero influence.
    filter_type: A string, one of 'linear', 'circular', or 'gaussian'.
    eps: A float, small value to avoid division by zero added to the entries.

  Returns: A BCOO sparse matrix of size (num_pts, num_pts) of the filter.
  """
  num_pts = coords.shape[0]

  distances = _cdist(coords, coords)

  row_indices, col_indices = jnp.where(distances <= cutoff_distance)
  relevant_distances = distances[row_indices, col_indices]

  if filter_type == Filters.LINEAR:
    filter_values = 1.0 - (relevant_distances / cutoff_distance)

  elif filter_type == Filters.CIRCULAR:
    filter_values = jnp.sqrt(1.0 - (relevant_distances / cutoff_distance) ** 2)

  elif filter_type == Filters.GAUSSIAN:
    sigma = cutoff_distance / 3.0
    filter_values = jnp.exp(-0.5 * (relevant_distances / sigma) ** 2)

  else:
    raise ValueError(f"Unsupported filter type: {filter_type.name}")

  row_sums = jnp.zeros(num_pts).at[row_indices].add(filter_values) + eps
  inv_row_sums = 1.0 / row_sums
  normalized_filter_values = filter_values * inv_row_sums[row_indices]

  return jax_sprs.BCOO(
    (normalized_filter_values, jnp.stack([row_indices, col_indices], axis=1)),
    shape=(num_pts, num_pts),
  )


def threshold_filter(density: jax.Array, beta: float, eta: float = 0.5) -> jax.Array:
  """Threshold project the density, pushing the values towards 0/1.

  Args:
    density: Array of size (N, ...) that are in [0,1] that contain the
      density of the elements.
    beta: Sharpness of projection (typically ~ 1-32). Larger value indicates
      a sharper projection.
    eta: Center value about which the values are projected.

  Returns: The thresholded density value array of size (N, ...) with the denisty values
    pushed towards 1/0.
  """
  v1 = jnp.tanh(eta * beta)
  nm = v1 + jnp.tanh(beta * (density - eta))
  dnm = v1 + jnp.tanh(beta * (1.0 - eta))
  return nm / dnm


def compute_point_indices_in_shape(
  coords: np.ndarray,
  shape: shapely.Geometry,
) -> np.ndarray:
  """Compute a Boolean mask of points strictly contained in a Shapely geometry.

  NOTE: The test is strict: points on the boundary return `False`.

  Args:
    coords: An array of point coordinates of shape `(num_pts, 2)`, where column 0 is x
      and column 1 is y coordinates.
    shape: Any valid Shapely geometry (Polygon, MultiPolygon,) against which containment
      is tested.

  Returns:
    A Boolean `np.ndarray` of shape `(num_pts,)`, where entry `i` is `True` iff
    `coords[i]` lies strictly inside `shape`.
  """
  return shapely.contains_xy(shape, coords[:, 0], coords[:, 1])


@chex.dataclass
class GaussQuad:
  """Gauss-Legendre quadrature points and weights.

  Attributes:
    points: An array of shape (num_points, num_dim) containing the quadrature points.
    weights: An array of shape (num_points,) containing the quadrature weights.
  """

  points: ArrayLike
  weights: ArrayLike

  @property
  def num_points(self) -> int:
    return int(self.points.shape[0])

  @property
  def num_dim(self) -> int:
    return int(self.points.shape[1])


def gauss_quadrature(num_gp_per_dim: int, num_dim: int) -> GaussQuad:
  """Gauss-Legendre quadrature on [-1, 1]^d.

  For shells, we typically model as 2D mid-surface with 1D thickness, so num_dim=1 or 2.

  For integration along the thickness, num_dim=1 and num_gp_per_dim=3 gives 6th-order
  accuracy. This should be sufficient for most materials, but increase if you have
  highly nonlinear through-thickness behavior.

  For integration over the mid-surface, num_dim=2 and num_gp_per_dim=2 gives 4 points,
  which is sufficient for integrating the polynomial shape functions of a bilinear
  quadrilateral element. Increase num_gp_per_dim for higher-order elements or highly
  curved geometry.

  Args:
    num_gp_per_dim: Number of gauss points per dimension.
    num_dim: Number of dimensions (1 for thickness, 2 for mid-surface). Must be in
      range (1, 3).

  Returns: A GaussQuad dataclass containing:
    - points: An array of shape (num_gp_per_dim^num_dim, num_dim) quadrature points.
    - weights: An array of shape (num_gp_per_dim^num_dim,) quadrature weights.
  """
  # Get 1D Gauss points and weights
  x, w = np.polynomial.legendre.leggauss(num_gp_per_dim)

  if num_dim == 1:
    points = jnp.asarray(x.reshape(-1, 1))
    weights = jnp.asarray(w)

  elif num_dim == 2:
    # Generate 2D points and weights as tensor products of 1D points and weights
    points = jnp.array([[x_i, x_j] for x_i in x for x_j in x])
    weights = jnp.array([w_i * w_j for w_i in w for w_j in w])

  elif num_dim == 3:
    # Generate 3D points and weights as tensor products of 1D points and weights
    points = jnp.array([[x_i, x_j, x_k] for x_i in x for x_j in x for x_k in x])
    weights = jnp.array([w_i * w_j * w_k for w_i in w for w_j in w for w_k in w])

  else:
    raise ValueError(f"Dimension must be in {1, 2, 3}, got {num_dim}.")

  return GaussQuad(points=points, weights=weights)


class StressType(enum.Enum):
  """Enumeration of supported continuum stress measures.

  Attributes:
    CAUCHY: True (Eulerian) Cauchy stress σ, force per unit deformed area.
    KIRCHHOFF: Kirchhoff (weighted Cauchy) stress τ = J σ.
    PK1: 1st Piola-Kirchhoff stress P, force per unit reference area
      (mixed two-point tensor).
    PK2: 2nd Piola-Kirchhoff stress S, force per unit reference area
      pulled back to the reference configuration (symmetric).
  """

  CAUCHY = enum.auto()
  KIRCHHOFF = enum.auto()
  PK1 = enum.auto()
  PK2 = enum.auto()


def _to_kirchhoff(
  stress: jnp.ndarray,
  stress_type: StressType,
  def_grad: jnp.ndarray,
) -> jnp.ndarray:
  """Convert any stress measure to Kirchhoff stress τ.

  Mathematical definitions:
    τ = J σ                (from Cauchy)
    τ = P Fᵀ               (from 1st PK)
    τ = F S Fᵀ             (from 2nd PK)
    τ = τ                  (identity, from Kirchhoff)

  Args:
    stress: A stress tensor of shape `(n, n)`.
    stress_type: Source stress tensor of `StressType`.
    def_grad: The deformation gradient of shape `(n, n)`.

  Returns:
    Kirchhoff stress τ of shape `(n, n)`.

  """
  det_def_grad = jnp.linalg.det(def_grad)
  match stress_type:
    case StressType.CAUCHY:
      # τ_ij = J σ_ij
      return det_def_grad * stress

    case StressType.KIRCHHOFF:
      return stress

    case StressType.PK1:
      # τ_ij = P_ik F_jk  (P Fᵀ)
      return jnp.einsum("ik, jk -> ij", stress, def_grad)

    case StressType.PK2:
      # τ_il = F_ij S_jk F_lk  (F S Fᵀ)
      return jnp.einsum("ij, jk, lk -> il", def_grad, stress, def_grad)

    case _:
      raise ValueError(f"Unknown stress type: {stress_type!r}")


def _from_kirchhoff(
  tau: jnp.ndarray,
  stress_type: StressType,
  def_grad: jnp.ndarray,
) -> jnp.ndarray:
  """Convert Kirchhoff stress τ to any target stress measure.

  Mathematical definitions:
    σ = J⁻¹ τ              (to Cauchy)
    τ = τ                  (identity, to Kirchhoff)
    P = τ F⁻ᵀ              (to 1st PK)
    S = F⁻¹ τ F⁻ᵀ          (to 2nd PK)

  Args:
    tau: Kirchhoff stress of shape `(n, n)`.
    stress_type: The target `StressType` to convert to.
    def_grad: The deformation gradient of shape `(n, n)`.

  Returns:
    Target stress tensor of shape `(n, n)`.
  """
  det_def_grad = jnp.linalg.det(def_grad)
  inv_def_grad = jnp.linalg.inv(def_grad)

  match stress_type:
    case StressType.CAUCHY:
      # σ_ij = J⁻¹ τ_ij
      return tau / det_def_grad

    case StressType.KIRCHHOFF:
      return tau

    case StressType.PK1:
      # P_ij = τ_ik (F⁻ᵀ)_kj = τ_ik F⁻¹_jk
      return jnp.einsum("ik,jk->ij", tau, inv_def_grad)

    case StressType.PK2:
      # S_il = F⁻¹_ij τ_jk F⁻¹_lk  (F⁻¹ τ F⁻ᵀ)
      return jnp.einsum("ij,jk,lk->il", inv_def_grad, tau, inv_def_grad)

    case _:
      raise ValueError(f"Unknown stress type: {stress_type!r}")


def stress_converter(
  stress: jnp.ndarray,
  from_stress: StressType,
  to_stress: StressType,
  def_grad: jnp.ndarray,
) -> jnp.ndarray:
  """Convert a stress tensor between any two continuum stress measures.

  Pipeline:
    stress  ──[_to_kirchhoff]──►  τ  ──[_from_kirchhoff]──►  result

  Supported conversions (row = source, col = target):

    ┌───────────┬──────────┬───────────┬──────────────┬────────────────┐
    │           │  Cauchy σ│ Kirchhoff │   1st PK P   │   2nd PK S     │
    ├───────────┼──────────┼───────────┼──────────────┼────────────────┤
    │  Cauchy σ │    –     │   J σ     │  J σ F⁻ᵀ     │ J F⁻¹ σ F⁻ᵀ    │
    │ Kirchhoff │  J⁻¹ τ   │    –      │   τ F⁻ᵀ      │ F⁻¹ τ F⁻ᵀ      │
    │  1st PK P │ J⁻¹PFᵀ   │  P Fᵀ     │    –         │   F⁻¹ P        │
    │  2nd PK S │J⁻¹FSFᵀ   │  F S Fᵀ   │    F S       │    –           │
    └───────────┴──────────┴───────────┴──────────────┴────────────────┘

  Args:
    stress: The input stress tensor of shape `(num_dim, num_dim)`.
    from_stress: The source `StressType` to convert from.
    to_stress: The target `StressType` to convert to.
    def_grad: Deformation gradient **F** of shape
      `(num_dim, num_dim)`, mapping reference to current configuration.

  Returns:
    Converted stress tensor of shape ``(num_dim, num_dim)``.
  """
  if from_stress is to_stress:
    return stress

  # Hub-and-spoke: stress → τ → target
  tau = _to_kirchhoff(stress, from_stress, def_grad)
  return _from_kirchhoff(tau, to_stress, def_grad)


def _skew(v: jnp.ndarray) -> jnp.ndarray:
  """Compute the 3x3 skew-symmetric matrix corresponding to a 3D vector.

  Constructs the matrix K such that K @ w = v × w for any vector w.

  Args:
    v: Array of (num_dim=3,) representing a vector in 3D Euclidean space.

  Returns:
    Array of (num_dim=3, num_dim=3) containing the skew-symmetric matrix of v.
  """
  return jnp.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def compute_rodrigues_matrix(theta: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
  """Compute the finite rotation matrix R(θ) ∈ SO(3) via the Rodrigues formula.

                            [   0     -𝜽_z     𝜽_y ]
                [𝜽]_×  =    [  𝜽_z      0     -𝜽_x ]
                            [ -𝜽_y     𝜽_x      0  ]

      R(θ) = I + (sin‖θ‖ / ‖θ‖) [θ]_× + ((1 - cos‖θ‖) / ‖θ‖²) [θ]_×²

  Uses Taylor series expansions for small angles to ensure safe JAX gradients.

  Args:
    theta: Array of (num_dim=3,) containing the rotation pseudo-vector. The
      direction defines the rotation axis and its norm defines the rotation
      angle in radians.
    eps: Scalar float threshold below which ‖θ‖² triggers the small-angle Taylor
      series approximation. Defaults to 1e-12.

  Returns:
    Array of (num_dim=3, num_dim=3) representing the rotation matrix R(θ).
  """
  t2 = jnp.dot(theta, theta)
  k = _skew(theta)
  k2 = jnp.einsum("ij, jk -> ik", k, k)

  small = t2 < eps
  t = safe_sqrt(t2)

  safe_t = jnp.where(small, 1.0, t)
  safe_t2 = jnp.where(small, 1.0, t2)

  poly_sin = 1.0 - t2 / 6.0 + t2**2 / 120.0
  poly_cos = 0.5 - t2 / 24.0 + t2**2 / 720.0

  c1 = jnp.where(small, poly_sin, jnp.sin(safe_t) / safe_t)
  c2 = jnp.where(small, poly_cos, (1.0 - jnp.cos(safe_t)) / safe_t2)

  return jnp.eye(3) + c1 * k + c2 * k2


def rotation_matrix_to_vector(
  rotation_matrix: jnp.ndarray,
  small_angle_eps: float = 1e-12,
  large_angle_eps: float = 1e-3,
) -> jnp.ndarray:
  """Compute the rotation pseudo-vector θ ∈ ℝ³ for R ∈ SO(3) via the matrix logarithm.

  This is the inverse of `compute_rodrigues_matrix`. It returns the vector whose
  direction is the axis of rotation and whose magnitude is the rotation angle in
  radians, with ‖θ‖ ∈ [0, π].

  The extraction uses three regimes, because the naive formula is singular at both
  ends of the angle range.

  1. Extract the rotation angle φ from the trace, tr(R) = 1 + 2 cos φ, clipping the
     argument of arccos into [-1, 1] for safety.
  2. Generic regime. The antisymmetric part satisfies R − Rᵀ = 2 sin(φ) [n̂]ₓ, so the
     axial vector `a` has magnitude 2 sin φ and θ = (φ / (2 sin φ)) a.
  3. Small-angle regime, φ² < `small_angle_eps`. Here sin φ → 0 and the scaling
     factor is replaced by its Taylor expansion, φ / (2 sin φ) ≈ 0.5 + φ²/12.
  4. Large-angle regime, π − φ < `large_angle_eps`. Here sin φ → 0 again and the
     antisymmetric part degenerates, so the axis cannot be recovered from it. The
     symmetric part is used instead, since (R + I)/2 → n̂ n̂ᵀ as φ → π. The column of
     R + I with the largest norm is normalised to give n̂ up to sign, and the sign is
     fixed by requiring consistency with the residual antisymmetric components.

  All three regimes are evaluated unconditionally and selected with `jnp.where`, so
  the function is safe under `jax.grad` and `jax.vmap`.

  Args:
    rotation_matrix: Array of shape `(num_dim=3, num_dim=3)`. A rotation R ∈ SO(3)
      satisfying det(R) = +1 and RᵀR = I.
    small_angle_eps: Scalar float threshold on φ² below which the small-angle Taylor
      branch is used. Defaults to 1e-12.
    large_angle_eps: Scalar float threshold on (π − φ) below which the symmetric-part
      branch is used. Defaults to 1e-3, which keeps the generic-branch error below
      roughly 1e-8 in float64.

  Returns:
    theta: Array of shape `(num_dim=3,)`. The rotation pseudo-vector, ‖θ‖ ∈ [0, π].
  """
  cos_phi = jnp.clip(0.5 * (jnp.trace(rotation_matrix) - 1.0), -1.0, 1.0)
  phi = jnp.arccos(jnp.clip(cos_phi, -1.0 + 1e-15, 1.0 - 1e-15))
  phi_sq = phi * phi

  axial_vec = jnp.array(
    [
      rotation_matrix[2, 1] - rotation_matrix[1, 2],
      rotation_matrix[0, 2] - rotation_matrix[2, 0],
      rotation_matrix[1, 0] - rotation_matrix[0, 1],
    ]
  )

  is_small = phi_sq < small_angle_eps
  is_large = (jnp.pi - phi) < large_angle_eps

  safe_sin = jnp.where(is_small | is_large, 1.0, jnp.sin(phi))
  generic_coeff = jnp.where(is_small, 0.5 + phi_sq / 12.0, phi / (2.0 * safe_sin))
  theta_generic = generic_coeff * axial_vec

  symmetric_part = 0.5 * (rotation_matrix + jnp.eye(3))
  column_norms = jnp.linalg.norm(symmetric_part, axis=0)
  best_column = jnp.argmax(column_norms)
  axis_unnormalised = symmetric_part[:, best_column]
  axis = safe_normalize(axis_unnormalised, axis=-1)
  axis_sign = jnp.where(jnp.einsum("i, i ->", axis, axial_vec) < 0.0, -1.0, 1.0)
  theta_large = phi * axis_sign * axis

  return jnp.where(is_large, theta_large, theta_generic)


def interpolate_directors_geodesic_so3(
  directors_i: jnp.ndarray, directors_ip1: jnp.ndarray, t: float = 0.5
) -> jnp.ndarray:
  """Geodesic interpolation on SO(3) between two director frames at parameter t ∈ [0,1].

  Computes the intermediate material frame lying `t` fraction along the shortest
  curved path (geodesic) between the start frame and the end frame. t=0 recovers
  `directors_i` exactly, t=1 recovers `directors_ip1` exactly, t=0.5 gives the
  Riemannian midpoint. The result always has mutually orthonormal rows and det = +1.

  Both inputs store directors as rows: `directors[α, i]` is the i-th spatial
  component of director α (α=0 → d₁, α=1 → d₂, α=2 → d₃).

  1. Compute Relative Rotation (r_rel):
    We compute the rotation matrix that maps the material axes of frame `i` to the
    material axes of frame `i+1`. We calculate the dot products between all pairs of
    incoming and outgoing directors.
  2. Extract Rotation Vector (θ):
    We convert this relative rotation matrix `r_rel` into an axis-angle pseudo-vector
    using `rotation_matrix_to_vector`.
  3. Scale Rotation and Re-exponentiate (r_t):
    We scale the rotation vector by fraction `t`, returning it to the mtrx domain. This
    gives us the partial rotation matrix.
  4. Apply Partial Rotation to Start Frame:
    Finally, we apply this intermediate relative rotation back to our starting directors
    `directors_i` to obtain the interpolated frame in Euclidean space.


  Args:
    directors_i: Array of shape (num_dim=3, num_dim=3). Frame at node i,
      rows = d₁, d₂, d₃. Must satisfy `directors_i @ directors_i.T ≈ I` and det ≈ +1.
    directors_ip1: Array of shape (num_dim=3, num_dim=3). Frame at node i+1.
    t: Scalar float ∈ [0, 1]. Geodesic interpolation parameter.

  Returns:
    directors_t: Array of shape (num_dim=3, num_dim=3). Interpolated frame with
      orthonormal rows.
  """
  r_rel = jnp.einsum("ia, ja -> ij", directors_i, directors_ip1)
  r_t = compute_rodrigues_matrix(t * rotation_matrix_to_vector(r_rel))
  return jnp.einsum("ia, ij -> ja", directors_i, r_t)


def rotate_directors(rotation: jnp.ndarray, ref_directors: jnp.ndarray) -> jnp.ndarray:
  """Apply an incremental rotation to a reference director frame.

  Reconstructs the current material frame from the rotational DOF vector
  and the reference configuration directors. This is the forward map from
  DOF space to SO(3):

      Q_current = R(θ) · Q_ref
      directors_current = ref_directors · R(θ)ᵀ

  where R = `compute_rodrigues_matrix(rotation)` and Q = directors.T
  (directors stored as rows, rotation matrix has directors as columns).

  At rotation = 0: R = I → directors_current = ref_directors.

  Args:
    rotation: Array of shape (num_dim=3,) of the rotation pseudovector from the
       reference configuration [rad].
    ref_directors: Array of shape (num_direc=3, num_dim=3) of the reference director
      frame. Must satisfy orthonormality and det = +1.

  Returns:
    directors_current: Array of shape (num_direc=3, num_dim=3) of the rotated director
      frame with orthonormal rows and det = +1.
  """
  r_inc = compute_rodrigues_matrix(rotation)
  return jnp.einsum("kj, ij -> ki", ref_directors, r_inc)


def compute_rotation_from_rotated_directors(
  directors_current: jnp.ndarray, ref_directors: jnp.ndarray
) -> jnp.ndarray:
  """Extract the rotation vector from two director frames.

  Inverse of `rotate_directors`. Given the current and reference material
  frames, returns θ_inc such that:

      rotate_directors(θ, ref_directors) = directors_current
  Derivation:
      Q_current^T = R @ Q_ref^T
      →  Q_current^T @ Q_ref = R @ Q_ref^T @ Q_ref = R
      →  R(θ) =  Q_current^T · Q_ref

  At directors_current = ref_directors: R_inc = I → θ_inc = 0.

  Args:
    directors_current: Array of shape (3, 3). Current director frame, rows =
      d₁, d₂, d₃.
    ref_directors: Array of shape (3, 3). Reference director frame, same
      layout.

  Returns:
    incremental_rotation: Array of shape (3,). Incremental rotation pseudo-
      vector with ‖θ‖ ∈ [0, π).
  """
  r_inc = jnp.einsum("ki,kj->ij", directors_current, ref_directors)
  return rotation_matrix_to_vector(r_inc)


def wrap_rotation_vector(
  rotation_vector: jnp.ndarray,
  eps: float = 1.0e-12,
) -> jnp.ndarray:
  """Reduce a rotation pseudo-vector to the equivalent rotation with ‖θ‖ ≤ π.

  A rotation pseudo-vector encodes a rotation of ‖θ‖ radians about the axis θ/‖θ‖.
  The map θ ↦ R(θ) is 2π-periodic along any fixed ray, so infinitely many vectors
  describe the same rotation. This returns the one of smallest magnitude.

  The magnitude is wrapped into [−π, π] by subtracting the nearest integer multiple
  of 2π, and the vector is then rescaled to that magnitude while keeping its
  direction:

      φ         = ‖θ‖
      φ_wrapped = φ − 2π · round(φ / 2π)     ∈ [−π, π]
      θ_wrapped = θ · (φ_wrapped / φ)

  A negative φ_wrapped flips the sign of the vector, which reverses the axis while
  reducing the angle, and describes the same rotation. Since only φ changes and only
  by a multiple of 2π, R(θ_wrapped) = R(θ) exactly.

  The magnitude is floored at `eps` before the division so the zero vector maps to
  itself instead of producing a NaN.

  Args:
    rotation_vector: Array of shape `(..., num_dim=3)` of rotation pseudo-vectors of
      arbitrary magnitude. Any number of leading batch axes is accepted.
    eps: Small positive float flooring the magnitude before division, guarding the
      near-zero case.

  Returns:
    Array of the same shape, satisfying ‖θ‖ ≤ π and encoding the identical rotation.
  """
  angle = jnp.linalg.norm(rotation_vector, axis=-1, keepdims=True)
  wrapped_angle = angle - 2.0 * jnp.pi * jnp.round(angle / (2.0 * jnp.pi))
  return rotation_vector * (wrapped_angle / jnp.maximum(angle, eps))


def polar_azith_to_vector(
  polar_ang: jnp.ndarray,
  azith_ang: jnp.ndarray,
  vec_mag: jnp.ndarray = 1.0,
  ang_units: Literal["radians", "degrees"] = "radians",
) -> jnp.ndarray:
  """Convert polar and azimuthal angles to a 3D vector.

  The polar angle is measured from the positive z-axis, and the azimuthal angle is
  measured from the positive x-axis in the x-y plane. The resulting vector has a
  magnitude of `vec_mag` (defaults to unit vector) with the directions:
    x = vec_mag * sin(polar) * cos(azith)
    y = vec_mag * sin(polar) * sin(azith)
    z = vec_mag * cos(polar)

  Args:
    polar_ang: Array of shape `(...)` of polar angles in radians or degrees.
    azith_ang: Array of shape `(...)` of azimuthal angles in radians or degrees.
    vec_mag: Array of shape `(...)` of the magnitudes of the resulting vectors.
      Defaults to 1.0 (unit vectors).
    ang_units: String indicating the units of the input angles. Can be either
      "radians" or "degrees". Defaults to "radians".

  Returns:
    Array of shape `(..., num_dim=3)` of the Cartesian vectors corresponding to
      the given angles and magnitudes. Scalar inputs give shape `(3,)`.
  """
  if ang_units == "degrees":
    polar_ang, azith_ang = jnp.deg2rad(polar_ang), jnp.deg2rad(azith_ang)

  rad_proj = vec_mag * jnp.sin(polar_ang)
  return jnp.stack(
    [
      rad_proj * jnp.cos(azith_ang),
      rad_proj * jnp.sin(azith_ang),
      vec_mag * jnp.cos(polar_ang),
    ],
    axis=-1,
  )


def vector_to_polar_azith(
  vec: jnp.ndarray, eps: float = 1.0e-12
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Decompose a 3D vector into polar angle, azimuthal angle, and magnitude.

  This is the inverse of `_utils.polar_azith_to_vector`. With the polar angle `θ`
  measured from the positive z axis and the azimuthal angle `φ` measured from the
  positive x axis in the x-y plane,

      ‖v‖ = √(v₁² + v₂² + v₃²),   θ = arccos(v₃ / ‖v‖),   φ = atan2(v₂, v₁).

  The magnitude is floored at `eps` before the division so that the zero vector
  returns a finite polar angle rather than a NaN.

  The decomposition is elementwise over the leading axes: the trailing axis of
  size 3 is contracted away, so each output has one axis fewer than `vec`.

  Args:
    vec: Array of shape `(..., num_dim=3)` of the Cartesian vectors.
    eps: Scalar float flooring the magnitude before division, guarding the polar
      angle against the zero vector. Defaults to 1e-12.

  Returns:
    A tuple `(polar_ang, azith_ang, vec_mag)`. Each entry is an array of shape
      `(...)`, that is, the shape of `vec` without its trailing axis. The polar
      angle lies in `[0, π]`, the azimuthal angle in `[-π, π]`, and the
      magnitude is non-negative.
  """
  vec_mag = jnp.linalg.norm(vec, axis=-1)
  safe_mag = jnp.maximum(vec_mag, eps)
  polar_ang = jnp.arccos(jnp.clip(vec[..., 2] / safe_mag, -1.0, 1.0))
  azith_ang = jnp.arctan2(vec[..., 1], vec[..., 0])
  return polar_ang, azith_ang, vec_mag


def affine_transform(
  coords: jnp.ndarray,
  translation: jnp.ndarray,
  rotation_matrix: jnp.ndarray,
  direction: Literal["LOCAL_TO_GLOBAL", "GLOBAL_TO_LOCAL"],
) -> jnp.ndarray:
  """Map point coordinates between a local frame and a global frame.

  A local frame is placed in the global frame by a rotation `R` and a translation
  `T`, so a point with local coordinates `x_local` has global coordinates

      x_global = R x_local + T.

  Inverting gives the map in the opposite direction,

      x_local = Rᵀ (x_global − T),

  where the translation is subtracted before the rotation is applied.

  NOTE: This maps points in space. Direction-valued quantities such as gradients,
  tangents, directors and field vectors rotate but do not translate, so they must not
  be passed here with a non-zero translation. Rotate them directly, or call this with
  a zero translation.

  Args:
    coords: Array of shape `(..., num_dim)` of point coordinates in the frame implied
      by `direction`.
    translation: Array of shape `(num_dim,)`. Position of the local frame origin
      expressed in global coordinates.
    rotation_matrix: Array of shape `(num_dim, num_dim)` in SO(d). Carries local
      frame directions into the global frame, so its columns are the local basis
      vectors written in global coordinates.
    direction: Which way to map. `LOCAL_TO_GLOBAL` applies `R x + T`,
      `GLOBAL_TO_LOCAL` applies `Rᵀ (x − T)`.

  Returns:
    Array of the same shape as `coords`, expressed in the target frame.
  """
  if direction == "LOCAL_TO_GLOBAL":
    return jnp.einsum("ij, ...j -> ...i", rotation_matrix, coords) + translation

  if direction == "GLOBAL_TO_LOCAL":
    return jnp.einsum("ji, ...j -> ...i", rotation_matrix, coords - translation)

  raise ValueError(f"{direction} must be LOCAL_TO_GLOBAL or GLOBAL_TO_LOCAL")
