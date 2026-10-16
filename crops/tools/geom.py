"""Geometry helpers for rod."""

import numpy as np


def annular_fiber_offsets(
  num_elems: int,
  num_fibers: int,
  semi_axis_d1: float,
  semi_axis_d2: float | None = None,
  norm_exponent: float = 2.0,
) -> np.ndarray:
  """Distributes fibers uniformly around a closed superelliptical cross-sectional ring.

  The fibers are placed on the curve

      |y₁ / a₁|ⁿ + |y₂ / a₂|ⁿ = 1,

  expressed in the local director basis (d₁, d₂), where a₁ and a₂ are the semi axes
  along d₁ and d₂ and n is the norm exponent. The curve is sampled at the angles
  θⱼ = 2πj / num_fibers for j = 0, …, num_fibers − 1, through the parameterization

      y₁(θ) = a₁ sgn(cos θ) |cos θ|^(2/n),
      y₂(θ) = a₂ sgn(sin θ) |sin θ|^(2/n),

  The exponent selects the family of sections. Setting n = 2 recovers an ellipse, and
  additionally setting a₁ = a₂ recovers a circle of radius a₁. Setting n = 1 gives a
  rhombus with vertices on the two axes. Increasing n beyond two progressively
  squares off the corners, approaching a rectangle of half widths a₁ and a₂.

  Args:
    num_elems: Number of rod elements. The identical ring is repeated on every
      element, so this argument only sets the leading dimension of the result.
    num_fibers: Number of fibers placed around the ring. The endpoint is excluded so
      that the ring closes without a duplicated fiber at θ = 2π.
    semi_axis_d1: Scalar float [m]. The semi axis a₁ along the director d₁, strictly
      positive.
    semi_axis_d2: Scalar float [m]. The semi axis a₂ along the director d₂, strictly
      positive. Defaults to `None`, in which case it is set equal to `semi_axis_d1`
      and the ring becomes d₁ d₂ symmetric.
    norm_exponent: Scalar float, the exponent n, strictly positive and typically in
      (0, 20]. Values far above twenty are numerically unreliable, because the
      exponent 2/n approaches zero and the floating point value of cos θ near π/2 is
      then raised to a power small enough that it no longer maps to zero.

  Returns:
    Array of shape `(num_elems, num_fibers, 2)` [m] holding the offset of every fiber
      from the centerline, with the trailing axis ordered as the components along d₁
      and d₂.
  """
  if semi_axis_d2 is None:
    semi_axis_d2 = semi_axis_d1

  angles = np.linspace(0.0, 2.0 * np.pi, num_fibers, endpoint=False)
  cos_angles, sin_angles = np.cos(angles), np.sin(angles)
  shape_exp = 2.0 / norm_exponent

  offsets_d1 = semi_axis_d1 * np.sign(cos_angles) * np.abs(cos_angles) ** shape_exp
  offsets_d2 = semi_axis_d2 * np.sign(sin_angles) * np.abs(sin_angles) ** shape_exp
  offsets_local = np.stack([offsets_d1, offsets_d2], axis=-1)

  return np.tile(offsets_local, (num_elems, 1, 1))


def create_stent_mask(
  unit_cell: np.ndarray,
  num_elems: int,
  num_fibers: int,
  num_cells_axial: int,
  num_cells_circum: int,
  area_floor: float = 1e-3,
  solid_is_light: bool = True,
) -> np.ndarray:
  """Converts a unit cell image into a per fiber material presence mask for a tube.

  The rod represents the stent as a whole, so its cross-section is the annular wall
  of the tube. The fibers are circumferential stations around that wall and the
  elements are axial stations along it, so the pair `(element, fiber)` indexes the
  unrolled cylindrical surface directly and a strut pattern drawn on a flat sheet
  maps onto it with no further transformation.

  The unit cell is binarized, resampled to the `(num_fibers / num_cells_circum,
  num_elems / num_cells_axial)` samples that one cell is allotted on that grid, and
  then tiled. Sampling is nearest neighbour at the centre of each sample, so a strut
  narrower than one sample may vanish. Raising the number of samples per cell is the
  remedy.

  The mask is intended to multiply the fiber areas and second moments, since a
  perforation removes material rather than softening it. Keeping it out of the
  pseudodensity leaves the pseudodensity free to describe the phase composition of
  the material that remains. Observe that `area_floor` is not cosmetic, because a
  pattern whose axial gaps span an entire ring leaves every fiber of one element
  void, that element then carries no stiffness, and the tangent operator becomes
  singular.

  Args:
    unit_cell: Array of shape `(num_pixels_circum, num_pixels_axial)` or with a
      trailing channel axis. Rows increase along the circumference and columns along
      the axis. Values may lie in `[0, 1]` or in `[0, 255]`.
    num_elems: Number of rod elements. Must be a multiple of `num_cells_axial`.
    num_fibers: Number of fibers. Must be a multiple of `num_cells_circum`.
    num_cells_axial: Number of cell repetitions along the tube, strictly positive.
    num_cells_circum: Number of cell repetitions around the tube, strictly positive.
    area_floor: Scalar float, `(float, (0, 1))`. Residual fraction retained where the
      pattern is void. Defaults to `1e-3`.
    solid_is_light: Whether light pixels denote strut material. Defaults to `True`.

  Returns:
    Array of shape `(num_elems, num_fibers)`, equal to one where strut material is
      present and to `area_floor` where it is absent.
  """
  assert num_fibers % num_cells_circum == 0, "cells must tile the fibers exactly."
  assert num_elems % num_cells_axial == 0, "cells must tile the elements exactly."

  cell = np.asarray(unit_cell, dtype=float)
  if cell.ndim == 3:
    cell = cell[..., :3].mean(axis=-1)
  solid = cell >= 0.5 * cell.max() if solid_is_light else cell < 0.5 * cell.max()

  num_rows, num_cols = num_fibers // num_cells_circum, num_elems // num_cells_axial
  rows = ((np.arange(num_rows) + 0.5) * solid.shape[0] / num_rows).astype(int)
  cols = ((np.arange(num_cols) + 0.5) * solid.shape[1] / num_cols).astype(int)

  tiled = np.tile(solid[np.ix_(rows, cols)], (num_cells_circum, num_cells_axial))

  return area_floor + (1.0 - area_floor) * tiled.T.astype(float)
