"""Visualization utilities for rods."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from skimage.measure import marching_cubes

import crops.src.utils as _utils

_DIRECTOR_STYLES = (
  ("red", "d₁ (In-plane transverse)"),
  ("green", "d₂ (Out-of-plane)"),
  ("blue", "d₃ (Tangent)"),
)

# Attribute stashed on an axis to accumulate the bounds of everything drawn on
# it, so that each plotting call can re-frame the viewport around the whole
# scene without the caller having to run a separate finalization step.
_BOUNDS_ATTR = "_rod_scene_bounds"


def _prepare_axes(ax: Optional[plt.Axes]) -> plt.Axes:
  """Return a clean 3D axis, instantiating a new figure canvas if needed.

  Args:
    ax: Optional pre-existing Matplotlib 3D axis context. If None, instantiates
      a clean new figure canvas and 3D projection axis.

  Returns:
    ax: A 3D axis with all background gridlines, panes, and canvas elements
      suppressed to emphasize the structural layout.
  """
  if ax is None:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

  ax.set_axis_off()
  return ax


def _extend_view(ax: plt.Axes, coords: np.ndarray, margin: float = 0.55) -> None:
  """Grow the axis' tracked scene bounds with `coords` and re-frame the viewport.

  Every drawing entry point calls this, so the viewport stays correctly framed
  around the union of everything plotted so far, regardless of call order.
  Viewport adjustments are fully coordinate-agnostic, tracking the structure
  dynamically across any spatial quadrant.

  Args:
    ax: Matplotlib 3D axis whose bounds are being extended.
    coords: Array of shape (..., num_dim=3) of coordinates just added to the scene.
    margin: Half-width of the viewport as a fraction of the largest data span.
  """
  coords = coords.reshape(-1, 3)
  s_min = np.min(coords, axis=0)
  s_max = np.max(coords, axis=0)

  tracked = getattr(ax, _BOUNDS_ATTR, None)
  if tracked is not None:
    s_min = np.minimum(s_min, tracked[0])
    s_max = np.maximum(s_max, tracked[1])
  setattr(ax, _BOUNDS_ATTR, (s_min, s_max))

  mid_point = (s_max + s_min) * 0.5
  half_width = np.max(s_max - s_min) * margin
  ax.set_xlim3d([mid_point[0] - half_width, mid_point[0] + half_width])
  ax.set_ylim3d([mid_point[1] - half_width, mid_point[1] + half_width])
  ax.set_zlim3d([mid_point[2] - half_width, mid_point[2] + half_width])


def _scene_bounds(ax: plt.Axes) -> tuple[np.ndarray, np.ndarray]:
  """Return the (min, max) corners of everything drawn on `ax` so far.

  Args:
    ax: Matplotlib 3D axis previously populated by one of the plotting functions.

  Returns:
    (s_min, s_max): Arrays of shape (num_dim=3,) of the scene's corner coordinates.

  Raises:
    ValueError: If nothing has been drawn on `ax` yet, so no bounds are known.
  """
  tracked = getattr(ax, _BOUNDS_ATTR, None)
  if tracked is None:
    raise ValueError(
      "No rod geometry has been drawn on this axis yet; plot the centerline or "
      "fibers before adding scene decorations."
    )
  return tracked


def _polyline_segments(points: np.ndarray) -> np.ndarray:
  """Convert consecutive points along axis 0 into individual line segments.

  Args:
    points: Array of shape (num_nodes, num_dim=3) for a single polyline, or
      (num_nodes, num_lines, num_dim=3) for a bundle of parallel polylines.

  Returns:
    segments: Array of shape (num_segments, 2, num_dim=3). For a bundle, segments
      are ordered element-major, i.e. index `e * num_lines + l`, matching the
      flattening of per-element/per-fiber field and area arrays.
  """
  return np.stack([points[:-1], points[1:]], axis=-2).reshape(-1, 2, 3)


def _linewidths_from_areas(
  areas: Optional[np.ndarray],
  linewidth_range: tuple[float, float],
  default: float,
) -> np.ndarray | float:
  """Map cross-sectional areas to line thicknesses via their square roots.

  Args:
    areas: Optional array of cross-sectional areas, in the same flattened order as
      the segments they belong to. If None, all segments share `default`.
    linewidth_range: Tuple of (min, max) linewidths spanned by the mapping.
    default: Uniform linewidth used when areas are absent or degenerate.

  Returns:
    linewidths: Per-segment linewidths, or `default` if no scaling applies.
  """
  if areas is None:
    return default

  areas = np.ravel(areas)
  max_area = np.max(areas)
  if max_area <= 0:
    return default

  lw_min, lw_max = linewidth_range
  return lw_min + (lw_max - lw_min) * np.sqrt(areas / max_area)


def _add_segments(
  ax: plt.Axes,
  segments: np.ndarray,
  field: Optional[np.ndarray],
  field_val_range: Optional[tuple[float, float]],
  areas: Optional[np.ndarray],
  cmap: str,
  base_color: list[float],
  default_linewidth: float,
  linewidth_range: tuple[float, float],
  plot_colorbar: bool,
  colorbar_label: str,
  colorbar_pad: float,
) -> None:
  """Draw 3D line segments, optionally scalar color-mapped and area-scaled.

  Shared by the centerline and the internal fibers, which differ only in their
  data arrays and cosmetic defaults.

  Args:
    ax: Matplotlib 3D axis to draw into.
    segments: Array of shape (num_segments, 2, num_dim=3).
    field: Optional scalar field per segment, flattened before use. If None, the
      segments are drawn with the flat `base_color`.
    field_val_range: Optional tuple of (min, max) color-map limits. If None,
      automatically computes local min/max from `field`.
    areas: Optional cross-sectional areas per segment used to scale thicknesses.
    cmap: String identifier of the Matplotlib colormap used when `field` is given.
    base_color: RGBA color used when `field` is None.
    default_linewidth: Uniform linewidth used when `areas` is None.
    linewidth_range: Tuple of (min, max) linewidths spanned when `areas` is given.
    plot_colorbar: Whether to display a colorbar legend for this field.
    colorbar_label: Label attached to this field's colorbar.
    colorbar_pad: Padding between the axis and this field's colorbar.
  """
  if field is not None:
    field = np.ravel(field)
    vmin, vmax = (
      field_val_range if field_val_range is not None else (np.min(field), np.max(field))
    )
    norm = Normalize(vmin=vmin, vmax=vmax)

    lc = Line3DCollection(segments, cmap=plt.get_cmap(cmap))
    lc.set_norm(norm)
    lc.set_array(field)

    if plot_colorbar:
      mappable = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
      mappable.set_array(field)
      ax.figure.colorbar(
        mappable, ax=ax, shrink=0.6, pad=colorbar_pad, aspect=20, label=colorbar_label
      )
  else:
    lc = Line3DCollection(segments)
    lc.set_color(base_color)

  lc.set_linewidths(_linewidths_from_areas(areas, linewidth_range, default_linewidth))
  ax.add_collection3d(lc)


def plot_rod_centerline(
  node_coords: np.ndarray,
  elem_areas: Optional[np.ndarray] = None,
  elem_field: Optional[np.ndarray] = None,
  elem_field_val_range: Optional[tuple[float, float]] = None,
  show_nodes: bool = False,
  cmap: str = "coolwarm",
  plot_colorbar: bool = False,
  colorbar_label: str = "Element Field",
  colorbar_pad: float = 0.05,
  ax: Optional[plt.Axes] = None,
) -> plt.Axes:
  """Render the discrete centerline segments of a Cosserat rod.

  Args:
    node_coords: Array of shape (num_nodes, num_dim=3) of global 3D Cartesian
      coordinates of the centerline nodes.
    elem_areas: Optional array of shape (num_elems,) of cross-sectional areas to
      scale element line thicknesses. If None, all elements are rendered with the
      same thickness.
    elem_field: Optional array of shape (num_elems,) of scalar variables to color-map
      the centerline elements. If None, a flat color is used.
    elem_field_val_range: Optional tuple of (min, max) bounds defining the color-map
      limits for `elem_field`. If None, automatically computes local min/max.
    show_nodes: Boolean flag indicating whether to mark discrete node centers with
      circle glyphs. Defaults to False.
    cmap: String identifier of the Matplotlib colormap utilized when scalar field
      tracking is active. Defaults to "coolwarm".
    plot_colorbar: Boolean flag indicating whether to display a colorbar legend when
      `elem_field` is given. Defaults to False.
    colorbar_label: Label attached to this field's colorbar.
    colorbar_pad: Padding between the axis and this field's colorbar. Bump this when
      stacking a second colorbar on the same axis.
    ax: Optional pre-existing Matplotlib 3D axis context. If None, instantiates a
      clean new figure canvas and 3D projection axis.

  Returns:
    ax: The active Matplotlib 3D Axes object, viewport re-framed around the scene.
  """
  ax = _prepare_axes(ax)

  _add_segments(
    ax,
    _polyline_segments(node_coords),
    field=elem_field,
    field_val_range=elem_field_val_range,
    areas=elem_areas,
    cmap=cmap,
    base_color=[0.2, 0.2, 0.8, 0.7],
    default_linewidth=3.0,
    linewidth_range=(1.5, 8.0),
    plot_colorbar=plot_colorbar,
    colorbar_label=colorbar_label,
    colorbar_pad=colorbar_pad,
  )

  if show_nodes:
    ax.scatter(
      node_coords[:, 0],
      node_coords[:, 1],
      node_coords[:, 2],
      color="black",
      marker="o",
      s=25,
      zorder=5,
    )

  _extend_view(ax, node_coords)
  return ax


def plot_rod_fibers(
  fiber_coordinates: np.ndarray,
  fiber_areas: Optional[np.ndarray] = None,
  fiber_field: Optional[np.ndarray] = None,
  fiber_field_val_range: Optional[tuple[float, float]] = None,
  cmap: str = "coolwarm",
  plot_colorbar: bool = False,
  colorbar_label: str = "Fiber Field",
  colorbar_pad: float = 0.05,
  ax: Optional[plt.Axes] = None,
) -> plt.Axes:
  """Render the internal sub-fiber profiles of a Cosserat rod.

  Args:
    fiber_coordinates: Array of shape (num_nodes, num_fibers, num_dim=3) specifying
      the global Cartesian coordinates of each internal sub-fiber segment.
    fiber_areas: Optional array of shape (num_elems, num_fibers) of cross-sectional
      areas for each individual sub-fiber segment to scale their line thicknesses.
    fiber_field: Optional array of shape (num_elems, num_fibers) of scalar variables
      to color-map across the internal fiber segments. If None, a flat color is used.
    fiber_field_val_range: Optional tuple of (min, max) bounds defining the color-map
      limits for `fiber_field`. If None, automatically computes local min/max.
    cmap: String identifier of the Matplotlib colormap utilized when scalar field
      tracking is active. Defaults to "coolwarm".
    plot_colorbar: Boolean flag indicating whether to display a colorbar legend when
      `fiber_field` is given. Defaults to False.
    colorbar_label: Label attached to this field's colorbar.
    colorbar_pad: Padding between the axis and this field's colorbar. Bump this when
      stacking a second colorbar on the same axis.
    ax: Optional pre-existing Matplotlib 3D axis context. If None, instantiates a
      clean new figure canvas and 3D projection axis.

  Returns:
    ax: The active Matplotlib 3D Axes object, viewport re-framed around the scene.
  """
  ax = _prepare_axes(ax)
  segments = _polyline_segments(fiber_coordinates)

  _add_segments(
    ax,
    segments,
    field=fiber_field,
    field_val_range=fiber_field_val_range,
    areas=fiber_areas,
    cmap=cmap,
    base_color=[0.3, 0.3, 0.3, 0.8],
    default_linewidth=2.0,
    linewidth_range=(0.5, 4.0),
    plot_colorbar=plot_colorbar,
    colorbar_label=colorbar_label,
    colorbar_pad=colorbar_pad,
  )

  _extend_view(ax, segments)
  return ax


def plot_rod_directors(
  node_coords: np.ndarray,
  directors: np.ndarray,
  arrow_scale: float = 0.06,
  plot_legend: bool = True,
  ax: Optional[plt.Axes] = None,
) -> plt.Axes:
  """Render the node-attached material frame triads of a Cosserat rod.

  Material Frame Axes:
    The local material frame {d₁, d₂, d₃} at each node is shown using an RGB
    color convention:
      - d₁ (Red): In-plane transverse director.
      - d₂ (Green): Out-of-plane transverse director.
      - d₃ (Blue): Unit tangent vector along the rod centerline.

  Args:
    node_coords: Array of shape (num_nodes, num_dim=3) of global 3D Cartesian
      coordinates of the centerline nodes the triads are anchored to.
    directors: Array of shape (num_nodes, num_dir=3, num_dim=3) of orthonormal
      reference frames. Row 0 is d₁, row 1 is d₂, row 2 is d₃.
    arrow_scale: Length of the rendered director arrows as a fraction of the largest
      span of the scene drawn so far. Defaults to 0.06.
    plot_legend: Boolean flag indicating whether to draw the RGB director legend.
      Defaults to True.
    ax: Optional pre-existing Matplotlib 3D axis context. If None, instantiates a
      clean new figure canvas and 3D projection axis.

  Returns:
    ax: The active Matplotlib 3D Axes object, viewport re-framed around the scene.
  """
  ax = _prepare_axes(ax)
  _extend_view(ax, node_coords)

  s_min, s_max = _scene_bounds(ax)
  arrow_len = np.max(s_max - s_min) * arrow_scale

  for idx, (color, _) in enumerate(_DIRECTOR_STYLES):
    ax.quiver(
      node_coords[:, 0],
      node_coords[:, 1],
      node_coords[:, 2],
      directors[:, idx, 0],
      directors[:, idx, 1],
      directors[:, idx, 2],
      color=color,
      length=arrow_len,
      normalize=True,
      linewidth=1.2,
    )

  if plot_legend:
    legend_elements = [
      Line2D([0], [0], color=c, lw=2, label=lbl) for c, lbl in _DIRECTOR_STYLES
    ]
    ax.legend(handles=legend_elements, loc="upper right", frameon=True)

  return ax


def plot_rod_bounding_box(ax: plt.Axes) -> plt.Axes:
  """Draw a semi-transparent cuboid around the physical domain with annotated corners.

  Bounds are taken from everything already drawn on `ax`, so call this after the
  centerline and fibers.

  Args:
    ax: Matplotlib 3D axis previously populated by one of the plotting functions.

  Returns:
    ax: The active Matplotlib 3D Axes object.
  """
  s_min, s_max = _scene_bounds(ax)
  x0, y0, z0 = s_min
  x1, y1, z1 = s_max

  v = [
    [x0, y0, z0],
    [x1, y0, z0],
    [x1, y1, z0],
    [x0, y1, z0],
    [x0, y0, z1],
    [x1, y0, z1],
    [x1, y1, z1],
    [x0, y1, z1],
  ]
  faces = [
    [v[0], v[1], v[2], v[3]],
    [v[4], v[5], v[6], v[7]],
    [v[0], v[1], v[5], v[4]],
    [v[2], v[3], v[7], v[6]],
    [v[0], v[3], v[7], v[4]],
    [v[1], v[2], v[6], v[5]],
  ]

  cuboid = Poly3DCollection(
    faces,
    facecolors=[0.4, 0.4, 0.4, 0.04],
    edgecolors=[0.2, 0.2, 0.2, 0.35],
    linewidths=0.8,
  )
  ax.add_collection3d(cuboid)

  for (x, y, z), tag in ((s_min, "Min"), (s_max, "Max")):
    ax.text(
      x,
      y,
      z,
      f"{tag}: [{x:.3f}, {y:.3f}, {z:.3f}]",
      color="black",
      fontsize=8,
      fontweight="bold",
    )

  return ax


def plot_rod_axis_triad(
  ax: plt.Axes, axis_scale: float = 0.15, offset_scale: float = 0.05
) -> plt.Axes:
  """Draw a labeled global XYZ reference triad beside the structure.

  The triad is anchored dynamically relative to the bounding data minimum of
  everything already drawn on `ax`, so call this after the centerline and fibers.

  Args:
    ax: Matplotlib 3D axis previously populated by one of the plotting functions.
    axis_scale: Length of each basis arrow as a fraction of the largest scene span.
    offset_scale: Distance the triad is pushed away from the scene's lower corner,
      as a fraction of the largest scene span.

  Returns:
    ax: The active Matplotlib 3D Axes object.
  """
  s_min, s_max = _scene_bounds(ax)
  max_range = np.max(s_max - s_min)

  origin = s_min - (max_range * offset_scale)
  axis_len = max_range * axis_scale
  basis_definitions = [
    ([axis_len, 0.0, 0.0], "X"),
    ([0.0, axis_len, 0.0], "Y"),
    ([0.0, 0.0, axis_len], "Z"),
  ]

  for vec, label in basis_definitions:
    ax.quiver(
      origin[0],
      origin[1],
      origin[2],
      vec[0],
      vec[1],
      vec[2],
      color="black",
      arrow_length_ratio=0.15,
      linewidth=2,
    )
    ax.text(
      origin[0] + vec[0] * 1.25,
      origin[1] + vec[1] * 1.25,
      origin[2] + vec[2] * 1.25,
      label,
      color="black",
      fontweight="bold",
    )

  return ax


# Cached marching-cubes output per obstacle name, so animation frames only upload
# transformed vertex positions rather than re-extracting the isosurface.
_OBSTACLE_MESH_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _polyline_edges(num_points: int, num_lines: int = 1) -> np.ndarray:
  """Build the edge connectivity of one or more parallel polylines.

  Polyscope curve networks are defined by a flat node array plus an edge index
  array, which is the connectivity analogue of the segment array that
  `_polyline_segments` builds for Matplotlib.

  Nodes are laid out line-major, so line `l` occupies indices
  `[l * num_points, (l + 1) * num_points)`. Edges are emitted in the same order,
  which matches the flattening of a per-fiber field transposed to `(num_fibers,
  num_elems)`. Note this is the opposite of the element-major ordering used by
  `_polyline_segments`, so per-fiber fields are transposed before upload here.

  Args:
    num_points: Number of points along each polyline.
    num_lines: Number of parallel polylines sharing this layout. Defaults to 1.

  Returns:
    Array of shape `(num_lines * (num_points - 1), 2)` of node index pairs.
  """
  base = np.stack([np.arange(num_points - 1), np.arange(1, num_points)], axis=1)
  return np.concatenate([base + line * num_points for line in range(num_lines)], axis=0)


def _node_radii_from_areas(
  areas: Optional[np.ndarray], num_nodes: int, default: float
) -> Optional[np.ndarray]:
  """Convert per-element cross-sectional areas to per-node equivalent radii.

  Elements carry areas but a curve network carries node radii, so interior nodes
  take the mean of their two adjacent elements and the two end nodes inherit their
  single neighbour. Each area is converted to the radius of a circle of that area,
  `r = sqrt(A / pi)`.

  Args:
    areas: Optional array of shape `(num_elems,)` [m^2]. `None` returns `None`.
    num_nodes: Number of centerline nodes, equal to `num_elems + 1`.
    default: Radius [m] substituted wherever the area is non-positive.

  Returns:
    Array of shape `(num_nodes,)` [m], or `None` if `areas` is `None`.
  """
  if areas is None:
    return None
  elem_areas = np.ravel(np.asarray(areas))
  node_areas = np.empty(num_nodes)
  node_areas[0] = elem_areas[0]
  node_areas[-1] = elem_areas[-1]
  node_areas[1:-1] = 0.5 * (elem_areas[:-1] + elem_areas[1:])
  return np.where(node_areas > 0.0, np.sqrt(node_areas / np.pi), default)


def elem_to_node_values(elem_values: np.ndarray) -> np.ndarray:
  """Averages per-element values onto the nodes of a rod.

  Interior nodes take the mean of their two adjacent elements and the two end nodes
  inherit their single neighbour, as in `_node_radii_from_areas`. Pass
  `rod.contact_radii` to get the node radii a curve network needs to render the rod
  at its contact envelope.

  Args:
    elem_values: Array of shape `(num_elems, ...)`.

  Returns:
    Array of shape `(num_elems + 1, ...)`.
  """
  elem_values = np.asarray(elem_values)
  interior = 0.5 * (elem_values[:-1] + elem_values[1:])
  return np.concatenate([elem_values[:1], interior, elem_values[-1:]], axis=0)


def _add_scalar_ps(
  structure,
  field: Optional[np.ndarray],
  field_val_range: Optional[tuple[float, float]],
  quantity_name: str,
  cmap: str,
) -> None:
  """Attach a per-edge scalar quantity to a Polyscope curve network.

  The Polyscope counterpart of the colour-mapping half of `_add_segments`. Does
  nothing when `field` is `None`, so callers need no branch.

  Args:
    structure: Registered Polyscope curve network.
    field: Optional scalar per edge, flattened before upload. Must already be in the
      edge ordering produced by `_polyline_edges`.
    field_val_range: Optional `(min, max)` colour-map limits. If `None`, the local
      minimum and maximum of `field` are used.
    quantity_name: Name of the scalar quantity in the Polyscope UI.
    cmap: Polyscope colormap name.
  """
  if field is None:
    return
  flat = np.ravel(field)
  vmin, vmax = (
    field_val_range
    if field_val_range is not None
    else (float(np.min(flat)), float(np.max(flat)))
  )
  structure.add_scalar_quantity(
    quantity_name,
    flat,
    defined_on="edges",
    cmap=cmap,
    vminmax=(vmin, vmax),
    enabled=True,
  )


def init_scene_ps(
  background_color: tuple[float, float, float, float] = (1, 1, 1, 1),
  window_size: Optional[tuple[int, int]] = None,
  ssaa_factor: Optional[int] = None,
  transparency_mode: Optional[str] = None,
):
  """Initialise Polyscope and set a clean background.

  The counterpart of `_prepare_axes`. Call once per session before any `_ps`
  plotting function; calling it again resets the scene appearance but leaves
  registered structures intact.

  The window may be larger than the screen. Screenshots are taken at the size of the
  framebuffer, which is twice the window size on a Retina display, so the window
  size sets the resolution of every render.

  Args:
    background_color: RGBA background. Defaults to opaque white.
    window_size: Optional `(width, height)` of the window in screen points.
    ssaa_factor: Optional supersampling factor, `2` or more smooths edges at the cost
      of rendering that many times larger internally.
    transparency_mode: Optional Polyscope transparency mode. `"pretty"` uses depth
      peeling, so overlapping translucent surfaces are ordered correctly.
  """
  if window_size is not None:
    ps.set_window_size(*window_size)
  ps.init()
  ps.set_background_color(background_color)
  ps.set_ground_plane_mode("none")
  if ssaa_factor is not None:
    ps.set_SSAA_factor(ssaa_factor)
  if transparency_mode is not None:
    ps.set_transparency_mode(transparency_mode)


def plot_rod_centerline_ps(
  node_coords: np.ndarray,
  elem_areas: Optional[np.ndarray] = None,
  elem_field: Optional[np.ndarray] = None,
  elem_field_val_range: Optional[tuple[float, float]] = None,
  show_nodes: bool = False,
  cmap: str = "coolwarm",
  name: str = "rod",
  base_color: tuple[float, float, float] = (0.2, 0.2, 0.8),
  default_radius: float = 0.002,
  node_radii: Optional[np.ndarray] = None,
):
  """Register the discrete centerline segments of a Cosserat rod in Polyscope.

  Polyscope counterpart of `plot_rod_centerline`. Colorbars are omitted because
  Polyscope draws them automatically for any enabled scalar quantity.

  When per-node radii are available the tube thickness varies along the rod, so a
  non-uniform cross-section renders as a varying profile rather than a uniform
  cylinder. Radii are taken from `node_radii` if given, otherwise derived from
  `elem_areas`, otherwise `default_radius` is used uniformly.

  Args:
    node_coords: Array of shape `(num_nodes, num_dim=3)` of global 3D Cartesian
      coordinates of the centerline nodes [m].
    elem_areas: Optional array of shape `(num_elems,)` of cross-sectional areas
      [m^2], converted to per-node equivalent radii.
    elem_field: Optional array of shape `(num_elems,)` of scalar variables to
      color-map the centerline elements. If None, a flat color is used.
    elem_field_val_range: Optional tuple of (min, max) color-map limits.
    show_nodes: Register an accompanying point cloud at the node positions.
    cmap: Polyscope colormap name used when `elem_field` is given.
    name: Polyscope structure name, the handle for later updates.
    base_color: RGB tube color used when `elem_field` is None.
    default_radius: Radius [m] used when neither radii source is available.
    node_radii: Optional array of shape `(num_nodes,)` [m] of explicit radii,
      overriding `elem_areas`. Pass `rod.contact_radii` interpolated to nodes to
      render the rod at the envelope the contact kernel actually sees.

  Returns:
    The registered Polyscope curve network.
  """
  num_nodes = node_coords.shape[0]
  curve = ps.register_curve_network(
    name, node_coords, _polyline_edges(num_nodes), color=base_color
  )
  radii = (
    node_radii
    if node_radii is not None
    else _node_radii_from_areas(elem_areas, num_nodes, default_radius)
  )
  if radii is None:
    curve.set_radius(default_radius, relative=False)
  else:
    curve.add_scalar_quantity("node_radius", np.asarray(radii), defined_on="nodes")
    curve.set_node_radius_quantity("node_radius", autoscale=False)

  _add_scalar_ps(curve, elem_field, elem_field_val_range, "elem_field", cmap)
  if show_nodes:
    ps.register_point_cloud(f"{name}_nodes", node_coords, radius=default_radius * 1.5)
  return curve


def plot_rod_fibers_ps(
  fiber_coordinates: np.ndarray,
  fiber_areas: Optional[np.ndarray] = None,
  fiber_field: Optional[np.ndarray] = None,
  fiber_field_val_range: Optional[tuple[float, float]] = None,
  cmap: str = "coolwarm",
  name: str = "rod_fibers",
  base_color: tuple[float, float, float] = (0.3, 0.3, 0.3),
  default_radius: float = 0.001,
):
  """Register the internal sub-fiber profiles of a Cosserat rod in Polyscope.

  Polyscope counterpart of `plot_rod_fibers`. Fiber nodes are laid out line-major
  to match `_polyline_edges`, so `fiber_field` is transposed before upload. That is
  the opposite of the element-major ordering used by the Matplotlib backend, but
  both accept the same `(num_elems, num_fibers)` input shape.

  Args:
    fiber_coordinates: Array of shape `(num_nodes, num_fibers, num_dim=3)` of global
      Cartesian coordinates of each internal sub-fiber node [m].
    fiber_areas: Optional array of shape `(num_elems, num_fibers)` of cross-sectional
      areas [m^2], converted to a single equivalent tube radius.
    fiber_field: Optional array of shape `(num_elems, num_fibers)` of scalar
      variables to color-map across the fiber segments. If None, a flat color.
    fiber_field_val_range: Optional tuple of (min, max) color-map limits.
    cmap: Polyscope colormap name used when `fiber_field` is given.
    name: Polyscope structure name.
    base_color: RGB tube color used when `fiber_field` is None.
    default_radius: Tube radius [m] used when `fiber_areas` is None.

  Returns:
    The registered Polyscope curve network.
  """
  num_nodes, num_fibers = fiber_coordinates.shape[0], fiber_coordinates.shape[1]
  fiber_nodes = fiber_coordinates.transpose(1, 0, 2).reshape(-1, 3)
  curve = ps.register_curve_network(
    name, fiber_nodes, _polyline_edges(num_nodes, num_fibers), color=base_color
  )

  if fiber_areas is None:
    curve.set_radius(default_radius, relative=False)
  else:
    # Per-fiber node radii, laid out line-major to match `_polyline_edges`.
    elem_radii = np.sqrt(np.maximum(np.asarray(fiber_areas), 0.0) / np.pi).T
    node_radii = np.empty((num_fibers, num_nodes))
    node_radii[:, 0], node_radii[:, -1] = elem_radii[:, 0], elem_radii[:, -1]
    node_radii[:, 1:-1] = 0.5 * (elem_radii[:, :-1] + elem_radii[:, 1:])
    node_radii = np.where(node_radii > 0.0, node_radii, default_radius)
    curve.add_scalar_quantity("node_radius", node_radii.ravel(), defined_on="nodes")
    curve.set_node_radius_quantity("node_radius", autoscale=False)

  _add_scalar_ps(
    curve,
    None if fiber_field is None else np.asarray(fiber_field).T,
    fiber_field_val_range,
    "fiber_field",
    cmap,
  )
  return curve


def plot_rod_directors_ps(
  node_coords: np.ndarray,
  directors: np.ndarray,
  arrow_scale: float = 0.06,
  name: str = "rod_directors",
  point_radius: float = 0.001,
):
  """Register the node-attached material frame triads of a Cosserat rod.

  Polyscope counterpart of `plot_rod_directors`. Uses the same RGB convention as
  `_DIRECTOR_STYLES`: d1 red, d2 green, d3 blue. Polyscope labels the quantities in
  its own UI, so there is no `plot_legend` analogue.

  Args:
    node_coords: Array of shape `(num_nodes, num_dim=3)` of centerline node
      coordinates the triads are anchored to [m].
    directors: Array of shape `(num_nodes, num_dir=3, num_dim=3)` of orthonormal
      frames. Row 0 is d1, row 1 is d2, row 2 is d3.
    arrow_scale: Arrow length as a fraction of the largest span of `node_coords`.
    name: Polyscope structure name.
    point_radius: Radius [m] of the anchor point cloud.

  Returns:
    The registered Polyscope point cloud carrying the three vector quantities.
  """
  span = float(np.max(np.ptp(node_coords, axis=0)))
  cloud = ps.register_point_cloud(name, node_coords, radius=point_radius)
  for idx, (color_name, _) in enumerate(_DIRECTOR_STYLES):
    rgb = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}
    cloud.add_vector_quantity(
      f"d{idx + 1}",
      directors[:, idx, :] * span * arrow_scale,
      color=rgb[color_name],
      vectortype="ambient",
      enabled=True,
    )
  return cloud


def plot_obstacle_ps(
  obstacle,
  step_idx: int = 0,
  name: str = "obstacle",
  base_color: tuple[float, float, float] = (0.76, 0.60, 0.42),
  transparency: float = 0.4,
):
  """Register an SDF obstacle as a Polyscope surface mesh.

  Extracts the zero level set with marching cubes in the obstacle body frame,
  caches the result under `name`, and registers it transformed into the lab frame
  by the pose at `step_idx`. The extraction is the expensive step and is done once;
  `update_obstacle_ps` reuses the cache for every subsequent frame.

  Args:
    obstacle: `contact.Obstacle` with a three-dimensional `sdf_grid`.
    step_idx: Load step index used to look up the pose via `obstacle.pose_at_step`.
      A static obstacle returns the identity pose for any index.
    name: Polyscope structure name, also the cache key.
    base_color: RGB surface color.
    transparency: Surface opacity in (0, 1].

  Returns:
    The registered Polyscope surface mesh.

  Raises:
    ValueError: If `obstacle.sdf_grid` is not three-dimensional, since marching
      cubes requires a volume.
  """
  if obstacle.sdf_grid.ndim != 3:
    raise ValueError(
      f"plot_obstacle_ps requires a 3D sdf_grid, got {obstacle.sdf_grid.ndim}D. "
      f"Plot a 2D obstacle with Matplotlib contours instead, or rebuild the field "
      f"on a 3D grid extruded along the unused axis."
    )

  verts, faces, _, _ = marching_cubes(
    volume=np.asarray(obstacle.sdf_grid),
    level=0.0,
    spacing=tuple(np.asarray(obstacle.spacing)),
  )
  verts = verts + np.asarray(obstacle.physical_origin)
  _OBSTACLE_MESH_CACHE[name] = (verts, faces)

  obj_trans, obj_rot_mat = obstacle.pose_at_step(step_idx)
  verts_glob = _utils.affine_transform(
    verts, np.asarray(obj_trans), np.asarray(obj_rot_mat), "LOCAL_TO_GLOBAL"
  )
  surface = ps.register_surface_mesh(
    name,
    verts_glob,
    faces,
    color=base_color,
  )
  surface.set_transparency(transparency)
  return surface


def update_rod_centerline_ps(
  node_coords: np.ndarray,
  elem_field: Optional[np.ndarray] = None,
  name: str = "rod",
) -> None:
  """Upload new centerline positions and field values for one animation frame.

  Args:
    node_coords: Updated node coordinates, shape `(num_nodes, num_dim=3)` [m].
    elem_field: Optional updated scalar per element, shape `(num_elems,)`. Pass
      `None` to leave the existing quantity untouched.
    name: Polyscope structure name used at registration.
  """
  curve = ps.get_curve_network(name)
  curve.update_node_positions(node_coords)
  if elem_field is not None:
    curve.add_scalar_quantity(
      "elem_field", np.ravel(elem_field), defined_on="edges", enabled=True
    )


def update_rod_fibers_ps(
  fiber_coordinates: np.ndarray,
  fiber_field: Optional[np.ndarray] = None,
  name: str = "rod_fibers",
) -> None:
  """Upload new fiber positions and field values for one animation frame.

  Args:
    fiber_coordinates: Updated fiber node coordinates, shape
      `(num_nodes, num_fibers, num_dim=3)` [m].
    fiber_field: Optional updated scalar, shape `(num_elems, num_fibers)`. Pass
      `None` to leave the existing quantity untouched.
    name: Polyscope structure name used at registration.
  """
  curve = ps.get_curve_network(name)
  curve.update_node_positions(fiber_coordinates.transpose(1, 0, 2).reshape(-1, 3))
  if fiber_field is not None:
    curve.add_scalar_quantity(
      "fiber_field",
      np.ravel(np.asarray(fiber_field).T),
      defined_on="edges",
      enabled=True,
    )


def update_rod_directors_ps(
  node_coords: np.ndarray,
  directors: np.ndarray,
  arrow_scale: float = 0.06,
  name: str = "rod_directors",
) -> None:
  """Upload new triad anchors and orientations for one animation frame.

  Args:
    node_coords: Updated node coordinates, shape `(num_nodes, num_dim=3)` [m].
    directors: Updated frames, shape `(num_nodes, num_dir=3, num_dim=3)`.
    arrow_scale: Arrow length as a fraction of the largest span of `node_coords`.
    name: Polyscope structure name used at registration.
  """
  span = float(np.max(np.ptp(node_coords, axis=0)))
  cloud = ps.get_point_cloud(name)
  cloud.update_point_positions(node_coords)
  rgb = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0)}
  for idx, (color_name, _) in enumerate(_DIRECTOR_STYLES):
    cloud.add_vector_quantity(
      f"d{idx + 1}",
      directors[:, idx, :] * span * arrow_scale,
      color=rgb[color_name],
      vectortype="ambient",
      enabled=True,
    )


def update_obstacle_ps(
  obstacle,
  step_idx: int,
  name: str = "obstacle",
) -> None:
  """Re-pose a registered obstacle for one animation frame.

  Reuses the cached marching-cubes result, so only transformed vertex positions are
  uploaded. No isosurface extraction and no re-registration.

  Args:
    obstacle: The same `contact.Obstacle` passed to `plot_obstacle_ps`.
    step_idx: Load step index used to look up the pose via `obstacle.pose_at_step`.
    name: Polyscope structure name used at registration.

  Raises:
    KeyError: If `plot_obstacle_ps` has not been called for this `name`.
  """
  if name not in _OBSTACLE_MESH_CACHE:
    raise KeyError(
      f"No cached obstacle mesh for '{name}'. Call plot_obstacle_ps first."
    )
  verts_body, _ = _OBSTACLE_MESH_CACHE[name]
  obj_trans, obj_rot_mat = obstacle.pose_at_step(step_idx)
  vert_glob = _utils.affine_transform(
    verts_body, np.asarray(obj_trans), np.asarray(obj_rot_mat), "LOCAL_TO_GLOBAL"
  )
  ps.get_surface_mesh(name).update_vertex_positions(vert_glob)


def look_at_best_fit_plane_ps(
  points: np.ndarray, fit_points: Optional[np.ndarray] = None, margin: float = 1.05
) -> None:
  """Points the camera at a point set along the normal of its best-fit plane.

  Suits a roughly planar curve, such as a vessel centerline, which is then seen
  face-on. The global axis closest to the points' principal direction is set as up, so
  the longest dimension runs vertically on screen. The camera aims at the middle of the
  on-screen bounding box of `fit_points` and sits just far enough back for all of them
  to fit the field of view of the current window, perspective included.

  Args:
    points: Array of shape `(num_points, num_dim=3)` [m] whose plane sets the view.
    fit_points: Optional array of shape `(num_fit_points, num_dim=3)` [m] that must
      all be in view, such as the vessel's vertices. Defaults to `points`.
    margin: Factor by which the framed points are shrunk inside the view, `1` or more.
  """
  center = points.mean(axis=0)
  _, _, principal = np.linalg.svd(points - center)
  up_axis = int(np.argmax(np.abs(principal[0])))
  normal = principal[2]
  screen_up = np.eye(3)[up_axis] - normal[up_axis] * normal
  screen_up /= np.linalg.norm(screen_up)
  screen_right = np.cross(screen_up, normal)

  fit_points = points if fit_points is None else fit_points
  offsets = fit_points - center
  target = center + sum(
    0.5 * (np.min(offsets @ axis) + np.max(offsets @ axis)) * axis
    for axis in (screen_right, screen_up)
  )
  offsets = fit_points - target
  # A point at depth `distance - z` is in view while |x|, |y| stay within the
  # half-width and half-height of the view at that depth.
  half_height = np.tan(0.5 * np.radians(ps.get_vertical_fov_degrees()))
  width, height = ps.get_window_size()
  half_width = half_height * width / height
  distance = np.max(
    margin
    * np.maximum(
      np.abs(offsets @ screen_right) / half_width,
      np.abs(offsets @ screen_up) / half_height,
    )
    + offsets @ normal
  )
  ps.set_up_dir(("x_up", "y_up", "z_up")[up_axis])
  ps.look_at(camera_location=tuple(target + distance * normal), target=tuple(target))


def top_left_anchor_ps(points: np.ndarray, radius: float) -> np.ndarray:
  """Returns the center of a ball placed beside the top-left of a point set on screen.

  The ball of `radius` sits half a radius left of the screen-space bounding box of
  `points`, with its top level with the top edge of the box, so it never covers them.
  The box is measured along the current camera's right and up directions, so set the
  view first.

  Args:
    points: Array of shape `(num_points, num_dim=3)` [m] the ball must not cover.
    radius: Ball radius [m].

  Returns:
    Array of shape `(num_dim=3,)` [m] of the ball center.
  """
  camera = ps.get_view_camera_parameters()
  screen_right = np.asarray(camera.get_right_dir())
  screen_up = np.asarray(camera.get_up_dir())
  center = points.mean(axis=0)
  offsets = points - center
  return (
    center
    + (np.min(offsets @ screen_right) - 1.5 * radius) * screen_right
    + (np.max(offsets @ screen_up) - radius) * screen_up
  )


def plot_direction_gizmo_ps(
  center: np.ndarray,
  radius: float,
  frame: np.ndarray,
  frame_colors: tuple[tuple[float, float, float], ...],
  name: str = "gizmo",
  ball_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
  ball_transparency: float = 0.25,
):
  """Registers a translucent ball with a reference frame drawn as thin axes.

  Together with `update_direction_gizmo_ps` it shows a direction, such as that of a
  uniform applied field, against the frame. Quantities take their structure's
  transparency, so the axes hang off a small opaque hub at the ball center rather
  than off the ball itself.

  Args:
    center: Array of shape `(num_dim=3,)` [m] of the ball center.
    radius: Ball radius [m]. The axes are `1.4 radius` long.
    frame: Array of shape `(num_axes, num_dim=3)` of unit axes, one per row.
    frame_colors: RGB color of each axis, in the row order of `frame`.
    name: Prefix of the two Polyscope structures, the handle for later updates.
    ball_color: RGB ball color.
    ball_transparency: Ball opacity in (0, 1].

  Returns:
    The registered hub point cloud that carries the axes.
  """
  ball = ps.register_point_cloud(
    f"{name}_ball", center[None], color=ball_color, transparency=ball_transparency
  )
  ball.set_radius(radius, relative=False)
  hub = ps.register_point_cloud(f"{name}_hub", center[None], color=(0.2, 0.2, 0.2))
  hub.set_radius(0.1 * radius, relative=False)
  for idx, (axis, color) in enumerate(zip(frame, frame_colors)):
    hub.add_vector_quantity(
      f"axis_{idx}",
      1.4 * radius * np.asarray(axis)[None],
      vectortype="ambient",
      enabled=True,
      radius=0.006,
      color=color,
    )
  return hub


def update_direction_gizmo_ps(
  direction: np.ndarray,
  length: float,
  name: str = "gizmo",
  color: tuple[float, float, float] = (0.0, 0.0, 0.0),
  arrow_radius: float = 0.012,
) -> None:
  """Draws a direction as an arrow from the gizmo center, for one animation frame.

  Only the direction is drawn, every arrow has the same `length`. A zero vector
  removes the arrow.

  Args:
    direction: Array of shape `(num_dim=3,)`, of any magnitude.
    length: Arrow length [m].
    name: Prefix given to `plot_direction_gizmo_ps`.
    color: RGB arrow color.
    arrow_radius: Arrow shaft radius, relative to the scene length scale.
  """
  hub = ps.get_point_cloud(f"{name}_hub")
  magnitude = float(np.linalg.norm(direction))
  if magnitude == 0.0:
    hub.remove_quantity("direction", error_if_absent=False)
    return
  hub.add_vector_quantity(
    "direction",
    length * np.asarray(direction)[None] / magnitude,
    vectortype="ambient",
    enabled=True,
    radius=arrow_radius,
    color=color,
  )


def plot_contact_spheres_ps(
  node_coords: np.ndarray,
  forces: np.ndarray,
  node_radii: np.ndarray,
  reference_force: float,
  max_radius: float,
  threshold: float = 0.05,
  cmap: Optional[Colormap] = None,
  transparency: float = 0.65,
  name: str = "contact",
):
  """Marks every rod node the wall pushes on with a translucent sphere.

  A node is in contact when the magnitude `f` of its wall force exceeds
  `threshold f_ref`. Its sphere sits on the rod surface facing the wall, one node
  radius from the node against the force the wall exerts, so the spheres trace the
  contact patches. Both the radius `max_radius √(f / f_ref)` and the color
  `cmap(f / f_ref)` encode the force, so the area a sphere covers is proportional to
  it. Forces above `f_ref` take the top color and grow past `max_radius`. Any
  structure already registered as `name` is removed first, so the function can be
  called once per animation frame.

  Args:
    node_coords: Array of shape `(num_nodes, num_dim=3)` [m] of the rod nodes.
    forces: Array of shape `(num_nodes, num_dim=3)` [N] of the force the wall exerts
      on each node, in the same frame as `node_coords`.
    node_radii: Array of shape `(num_nodes,)` [m] of the rod radius at each node.
    reference_force: Force `f_ref` [N] that maps to the top color and `max_radius`.
      Hold it fixed across frames and runs so they share one scale.
    max_radius: Sphere radius [m] at `f = f_ref`.
    threshold: Fraction of `f_ref` below which a node gets no sphere.
    cmap: Colormap of `f / f_ref` on `[0, 1]`. Defaults to `contact_force_cmap`.
    transparency: Sphere opacity in (0, 1].
    name: Polyscope structure name.

  Returns:
    The registered Polyscope point cloud, or `None` if no node is in contact.
  """
  cmap = contact_force_cmap if cmap is None else cmap
  ps.remove_point_cloud(name, error_if_absent=False)
  magnitude = np.linalg.norm(forces, axis=-1)
  in_contact = magnitude > threshold * reference_force
  if not in_contact.any():
    return None
  wall_side = -forces[in_contact] / magnitude[in_contact, None]
  centers = node_coords[in_contact] + node_radii[in_contact, None] * wall_side
  relative_force = magnitude[in_contact] / reference_force
  spheres = ps.register_point_cloud(name, centers, transparency=transparency)
  spheres.add_scalar_quantity("radius", max_radius * np.sqrt(relative_force))
  spheres.set_point_radius_quantity("radius", autoscale=False)
  spheres.add_color_quantity("force", cmap(relative_force)[:, :3], enabled=True)
  return spheres


def content_crop_box_ps(
  pad_fraction: float = 0.03, background_level: int = 250
) -> tuple[slice, slice]:
  """Returns the pixel box that holds everything drawn in the current view.

  Measure the box once, with the scene at its widest, and crop every later frame with
  it, so the frames of an animation stay aligned. Assumes a light background.

  Args:
    pad_fraction: Margin added on every side, as a fraction of the larger image side.
    background_level: Pixels with every RGB channel at or above this level count as
      background.

  Returns:
    A tuple `(rows, cols)` of slices that index a screenshot.
  """
  image = ps.screenshot_to_buffer(transparent_bg=False)
  ink = np.any(image[..., :3] < background_level, axis=-1)
  rows = np.flatnonzero(ink.any(axis=1))
  cols = np.flatnonzero(ink.any(axis=0))
  pad = int(pad_fraction * max(ink.shape))
  return (
    slice(max(rows[0] - pad, 0), rows[-1] + pad + 1),
    slice(max(cols[0] - pad, 0), cols[-1] + pad + 1),
  )


def screenshot_ps(crop_box: Optional[tuple[slice, slice]] = None) -> np.ndarray:
  """Renders the current view to an RGB image, optionally cropped.

  The image is as large as the framebuffer, so its resolution is set by the window
  size given to `init_scene_ps`.

  Args:
    crop_box: Optional `(rows, cols)` slices, as returned by `content_crop_box_ps`.

  Returns:
    Array of shape `(height, width, 3)` of `uint8` RGB values.
  """
  image = ps.screenshot_to_buffer(transparent_bg=False)[..., :3]
  return image if crop_box is None else image[crop_box]


_SECTION_CORNERS = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
_HEX_FACES = np.array(
  [[0, 1, 2, 3], [7, 6, 5, 4], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
)


def _fiber_section_hexes(
  fiber_coordinates: np.ndarray,
  directors: np.ndarray,
  fiber_heights: np.ndarray,
  fiber_thicknesses: np.ndarray,
) -> np.ndarray:
  """Build the six bounding faces of each element-fiber cell as a solid box.

  Every fiber segment is rendered as a hexahedron: the rectangular cross-section is
  laid out in the local `d1`-`d2` plane at both bounding nodes and the two faces are
  joined. Because the cross-section is carried by the node directors, each cell
  tilts and twists with the material frame rather than staying axis-aligned.

  Cross-section extents are constant within an element, which produces the stepped
  boundary between neighbouring elements rather than a smoothed one, matching the
  piecewise-constant nature of an element field.

  Args:
    fiber_coordinates: Array of shape `(num_nodes, num_fibers, num_dim=3)` [m] of
      global Cartesian fiber centreline coordinates at the nodes.
    directors: Array of shape `(num_nodes, num_dir=3, num_dim=3)` of orthonormal
      material frames. Row 0 is `d1` and row 1 is `d2`.
    fiber_heights: Array of shape `(num_elems, num_fibers)` [m] of each fiber's
      extent along `d1`.
    fiber_thicknesses: Array of shape `(num_elems, num_fibers)` [m] of each fiber's
      extent along `d2`.

  Returns:
    Array of shape `(num_elems, num_fibers, num_faces=6, num_corners=4, num_dim=3)`
      [m] of quadrilateral corner coordinates.
  """
  extents = np.stack([fiber_heights, fiber_thicknesses], axis=-1)
  frames = np.asarray(directors)[:, :2, :]

  # Cross-section corners of every element-fiber cell in the local d1-d2 plane. The
  # element's own extents are used at both of its bounding nodes, each carried by
  # that node's frame.
  local = np.einsum("ca, efa -> efca", _SECTION_CORNERS, extents)
  start = fiber_coordinates[:-1, :, None, :] + np.einsum(
    "efca, ead -> efcd", local, frames[:-1]
  )
  end = fiber_coordinates[1:, :, None, :] + np.einsum(
    "efca, ead -> efcd", local, frames[1:]
  )

  hexes = np.concatenate([start, end], axis=2)
  return hexes[:, :, _HEX_FACES, :]


def _annular_sector_face_indices(num_arc_segments: int) -> np.ndarray:
  """Quad connectivity of an annular-sector prism.

  The prism's corner points are assumed to be laid out as `(end, radius, arc)`,
  flattened in that order, with ends `{start, end}`, radii `{inner, outer}` and
  `num_arc_segments + 1` points along each arc. The faces are, in order: the outer
  wall, the inner wall, the start cap, the end cap and the two radial walls.

  Args:
    num_arc_segments: Number of quads used to approximate each arc.

  Returns:
    Integer array of shape `(num_faces=4 * num_arc_segments + 2, num_corners=4)`.
  """
  num_arc_points = num_arc_segments + 1

  def point(end, radius, arc):
    return (end * 2 + radius) * num_arc_points + arc

  arc = np.arange(num_arc_segments)
  nxt = arc + 1
  outer_wall = np.stack(
    [point(0, 1, arc), point(0, 1, nxt), point(1, 1, nxt), point(1, 1, arc)], axis=-1
  )
  inner_wall = np.stack(
    [point(0, 0, arc), point(1, 0, arc), point(1, 0, nxt), point(0, 0, nxt)], axis=-1
  )
  start_cap = np.stack(
    [point(0, 0, arc), point(0, 1, arc), point(0, 1, nxt), point(0, 0, nxt)], axis=-1
  )
  end_cap = np.stack(
    [point(1, 0, arc), point(1, 1, arc), point(1, 1, nxt), point(1, 0, nxt)], axis=-1
  )
  last = num_arc_segments
  radial_walls = np.array(
    [
      [point(0, 0, 0), point(0, 1, 0), point(1, 1, 0), point(1, 0, 0)],
      [point(0, 0, last), point(0, 1, last), point(1, 1, last), point(1, 0, last)],
    ]
  )
  return np.concatenate(
    [outer_wall, inner_wall, start_cap, end_cap, radial_walls], axis=0
  )


def _fiber_annular_sector_prisms(
  node_coords: np.ndarray,
  directors: np.ndarray,
  fiber_inner_radii: np.ndarray,
  fiber_outer_radii: np.ndarray,
  fiber_start_angles: np.ndarray,
  fiber_end_angles: np.ndarray,
  fiber_center_offsets: Optional[np.ndarray],
  num_arc_segments: int,
) -> np.ndarray:
  """Build the bounding faces of each element-fiber cell as an annular-sector prism.

  The annular-sector analogue of `_fiber_section_hexes`. Each cell is the region
  between two radii and two polar angles of the local `d1`-`d2` plane, extruded
  between the element's bounding nodes. Arcs are polygonised with
  `num_arc_segments` quads, and the sector parameters are constant within an
  element, so the rendering is stepped between neighbouring elements.

  Solid disks (`inner radius = 0`) and full rings (`end - start = 2π`) are
  expressed in the same layout. The inner wall of a disk collapses to zero area
  on its own; the two radial walls of a full ring coincide inside the solid, so
  they are collapsed here. Both are then dropped by `_flatten_cells`.

  Args:
    node_coords: Array of shape `(num_nodes, num_dim=3)` [m] of the rod centreline.
    directors: Array of shape `(num_nodes, num_dir=3, num_dim=3)` of orthonormal
      material frames. Row 0 is `d1` and row 1 is `d2`.
    fiber_inner_radii: Array of shape `(num_elems, num_fibers)` [m].
    fiber_outer_radii: Array of shape `(num_elems, num_fibers)` [m].
    fiber_start_angles: Array of shape `(num_elems, num_fibers)` [rad], measured
      in the cross-section plane from `d1` towards `d2`.
    fiber_end_angles: Array of shape `(num_elems, num_fibers)` [rad].
    fiber_center_offsets: Optional array of shape `(num_elems, num_fibers,
      num_crossec_dim=2)` [m] of each sector's centre in the `d1`-`d2` plane. If
      None, all sectors are centred on the rod centreline.
    num_arc_segments: Number of quads used to approximate each arc.

  Returns:
    Array of shape `(num_elems, num_fibers, num_faces=4 * num_arc_segments + 2,
      num_corners=4, num_dim=3)` [m] of quadrilateral corner coordinates.
  """
  num_elems, num_fibers = fiber_inner_radii.shape
  fractions = np.linspace(0.0, 1.0, num_arc_segments + 1)
  angles = (
    fiber_start_angles[..., None]
    + (fiber_end_angles - fiber_start_angles)[..., None] * fractions
  )
  radial = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (e, f, arc, 2)
  radii = np.stack([fiber_inner_radii, fiber_outer_radii], axis=-1)  # (e, f, 2)

  # In-plane offsets of every corner point, laid out as (radius, arc).
  local = radii[:, :, :, None, None] * radial[:, :, None, :, :]
  if fiber_center_offsets is not None:
    local = local + fiber_center_offsets[:, :, None, None, :]
  local = local.reshape(num_elems, num_fibers, -1, 2)

  frames = np.asarray(directors)[:, :2, :]
  start = node_coords[:-1, None, None, :] + np.einsum(
    "efpa, ead -> efpd", local, frames[:-1]
  )
  end = node_coords[1:, None, None, :] + np.einsum(
    "efpa, ead -> efpd", local, frames[1:]
  )

  points = np.concatenate([start, end], axis=2)  # (e, f, (end, radius, arc), 3)
  faces = points[:, :, _annular_sector_face_indices(num_arc_segments), :]

  # The radial walls are the last two faces; collapse them for full rings.
  full_ring = np.isclose(np.abs(fiber_end_angles - fiber_start_angles), 2.0 * np.pi)
  faces[full_ring, -2:, :, :] = faces[full_ring, -2:, :1, :]
  return faces


def _flatten_cells(
  cells: np.ndarray,
  field: Optional[np.ndarray],
  mask: Optional[np.ndarray],
) -> tuple[np.ndarray, Optional[np.ndarray]]:
  """Select the drawn cells, flatten their faces and align the field with them.

  Args:
    cells: Array of shape `(num_elems, num_fibers, num_faces, num_corners=4,
      num_dim=3)` [m] of per-cell face coordinates.
    field: Optional array of shape `(num_elems, num_fibers)` of per-cell scalars, or
      `(num_elems, num_fibers, 3 | 4)` of per-cell RGB(A) colours.
    mask: Optional boolean array of shape `(num_elems, num_fibers)`. Cells where it
      is False are dropped entirely.

  Returns:
    quads: Array of shape `(num_quads, num_corners=4, num_dim=3)` [m].
    values: Array of `num_quads` scalars, or `(num_quads, 3 | 4)` colours, aligned
      with `quads`; None if no field was given.
  """
  num_faces = cells.shape[2]
  keep = (
    np.ones(cells.shape[:2], dtype=bool) if mask is None else np.asarray(mask, bool)
  )
  quads = cells[keep].reshape(-1, 4, 3)
  values = (
    None if field is None else np.repeat(np.asarray(field)[keep], num_faces, axis=0)
  )

  # Drop zero-area faces, such as the collapsed inner wall of a solid disk or the
  # coincident radial walls of a full ring, so they neither draw stray edges nor
  # show through translucent neighbours.
  area = 0.5 * np.linalg.norm(
    np.cross(quads[:, 2] - quads[:, 0], quads[:, 3] - quads[:, 1]), axis=-1
  )
  nondegenerate = area > 1e-9 * np.max(area, initial=0.0)
  quads = quads[nondegenerate]
  if values is not None:
    values = values[nondegenerate]
  return quads, values


def _add_patches(
  ax: plt.Axes,
  quads: np.ndarray,
  field: Optional[np.ndarray],
  field_val_range: Optional[tuple[float, float]],
  cmap: str,
  alpha: float,
  edge_color,
  edge_width: float,
  plot_colorbar: bool,
  colorbar_label: str,
  colorbar_pad: float,
) -> None:
  """Attach a filled polygon collection to a 3D axis, optionally colour-mapped.

  The patch analogue of `_add_segments`. When `field` is `None` a flat facecolor is
  used, when it holds RGB(A) colours they are applied as they are, otherwise the
  values are mapped through `cmap` with either the supplied limits or the local
  minimum and maximum.

  Args:
    ax: Matplotlib 3D axis.
    quads: Array of shape `(num_quads, num_corners=4, num_dim=3)` [m].
    field: Optional array of `num_quads` scalars, flattened before use, or of shape
      `(num_quads, 3 | 4)` RGB(A) colours.
    field_val_range: Optional `(min, max)` colour-map limits.
    cmap: Matplotlib colormap name.
    alpha: Face opacity in `(0, 1]`.
    edge_color: Matplotlib colour for the polygon outlines.
    edge_width: Line width of the polygon outlines.
    plot_colorbar: Attach a colorbar when a field is given.
    colorbar_label: Label for the colorbar.
    colorbar_pad: Padding between the axis and the colorbar.
  """
  collection = Poly3DCollection(
    quads, alpha=alpha, edgecolors=edge_color, linewidths=edge_width
  )
  if field is None:
    collection.set_facecolor((0.55, 0.75, 0.95))
  elif np.ndim(field) == 2:
    collection.set_facecolor(np.asarray(field))
  else:
    flat = np.ravel(field)
    vmin, vmax = (
      field_val_range
      if field_val_range is not None
      else (float(np.min(flat)), float(np.max(flat)))
    )
    collection.set_array(flat)
    collection.set_cmap(cmap)
    collection.set_clim(vmin, vmax)
    if plot_colorbar:
      ax.figure.colorbar(collection, ax=ax, label=colorbar_label, pad=colorbar_pad)
  ax.add_collection3d(collection)


def plot_rod_fiber_sections(
  fiber_coordinates: np.ndarray,
  directors: np.ndarray,
  fiber_heights: np.ndarray,
  fiber_thicknesses: np.ndarray,
  fiber_field: Optional[np.ndarray] = None,
  fiber_field_val_range: Optional[tuple[float, float]] = None,
  fiber_mask: Optional[np.ndarray] = None,
  cmap: str = "coolwarm",
  alpha: float = 1.0,
  edge_color=(0.0, 0.0, 0.0),
  edge_width: float = 0.4,
  plot_colorbar: bool = False,
  colorbar_label: str = "Fiber Field",
  colorbar_pad: float = 0.05,
  ax: Optional[plt.Axes] = None,
) -> plt.Axes:
  """Render the fiber bundle as solid rectangular cross-sectional cells.

  Each element-fiber pair is drawn as a closed box rather than a line or a flat
  band, so a layered rectangular cross-section reads as a solid body with visible
  depth. Fibers offset along `d2` occupy distinct positions through the thickness
  and are handled without special casing, so a fully populated cross-sectional grid
  renders correctly. See `plot_rod_annular_sections` for circular layouts.

  For a side view, call `ax.view_init(elev=0, azim=-90)` on the returned axis.

  Args:
    fiber_coordinates: Array of shape `(num_nodes, num_fibers, num_dim=3)` [m] of
      global Cartesian fiber coordinates, as returned by
      `mesher.compute_fiber_positions`.
    directors: Array of shape `(num_nodes, num_dir=3, num_dim=3)` of orthonormal
      material frames at the nodes.
    fiber_heights: Array of shape `(num_elems, num_fibers)` [m] of each fiber's
      extent along `d1`.
    fiber_thicknesses: Array of shape `(num_elems, num_fibers)` [m] of each fiber's
      extent along `d2`.
    fiber_field: Optional array of shape `(num_elems, num_fibers)` of scalars used
      to colour each cell through `cmap`, or `(num_elems, num_fibers, 3 | 4)` of
      RGB(A) colours used as they are. If None, a flat facecolor is used.
    fiber_field_val_range: Optional `(min, max)` colour-map limits. If None, the
      local minimum and maximum are used.
    fiber_mask: Optional boolean array of shape `(num_elems, num_fibers)`. Cells
      where it is False are not drawn at all, so a thresholded pseudo-density
      (e.g. `density > 0.5`) removes void segments from the render rather than
      merely colouring them.
    cmap: Matplotlib colormap name or colormap instance.
    alpha: Face opacity in `(0, 1]`. Lower values let occluded cells show through,
      which helps when fibers are stacked along `d2`.
    edge_color: Matplotlib colour for the cell outlines.
    edge_width: Line width of the cell outlines. Set to 0 to hide them.
    plot_colorbar: Attach a colorbar when a field is given.
    colorbar_label: Label for this field's colorbar.
    colorbar_pad: Padding between the axis and the colorbar.
    ax: Optional pre-existing Matplotlib 3D axis. If None, one is created.

  Returns:
    ax: The Matplotlib 3D Axes, with the viewport re-framed around the bundle.
  """
  ax = _prepare_axes(ax)
  cells = _fiber_section_hexes(
    fiber_coordinates=np.asarray(fiber_coordinates),
    directors=np.asarray(directors),
    fiber_heights=np.asarray(fiber_heights),
    fiber_thicknesses=np.asarray(fiber_thicknesses),
  )
  quads, values = _flatten_cells(cells, fiber_field, fiber_mask)
  if quads.shape[0] == 0:
    return ax
  _add_patches(
    ax=ax,
    quads=quads,
    field=values,
    field_val_range=fiber_field_val_range,
    cmap=cmap,
    alpha=alpha,
    edge_color=edge_color,
    edge_width=edge_width,
    plot_colorbar=plot_colorbar,
    colorbar_label=colorbar_label,
    colorbar_pad=colorbar_pad,
  )
  _extend_view(ax, quads)
  return ax


def plot_rod_annular_sections(
  node_coords: np.ndarray,
  directors: np.ndarray,
  fiber_inner_radii: np.ndarray,
  fiber_outer_radii: np.ndarray,
  fiber_start_angles: np.ndarray,
  fiber_end_angles: np.ndarray,
  fiber_center_offsets: Optional[np.ndarray] = None,
  num_arc_segments: int = 8,
  fiber_field: Optional[np.ndarray] = None,
  fiber_field_val_range: Optional[tuple[float, float]] = None,
  fiber_mask: Optional[np.ndarray] = None,
  cmap: str = "coolwarm",
  alpha: float = 1.0,
  edge_color=(0.0, 0.0, 0.0),
  edge_width: float = 0.4,
  plot_colorbar: bool = False,
  colorbar_label: str = "Fiber Field",
  colorbar_pad: float = 0.05,
  ax: Optional[plt.Axes] = None,
) -> plt.Axes:
  """Render the fiber bundle as solid annular-sector cross-sectional cells.

  The circular analogue of `plot_rod_fiber_sections`. Each element-fiber pair is
  the sector of an annulus between `fiber_inner_radii` and `fiber_outer_radii`
  spanning `fiber_start_angles` to `fiber_end_angles` in the local `d1`-`d2`
  plane, extruded between the element's bounding nodes. Rings of sectors tile a
  full circular or tubular cross-section; a single fiber with inner radius 0 and a
  `2π` span is a solid round rod; and `fiber_center_offsets` lets round fibers sit
  off the centreline (e.g. a bundle of circular strands).

  Cells are anchored on the rod centreline rather than on fiber centroids, because
  a sector is defined relative to the rod axis.

  Args:
    node_coords: Array of shape `(num_nodes, num_dim=3)` [m] of the rod centreline,
      in the reference or deformed configuration.
    directors: Array of shape `(num_nodes, num_dir=3, num_dim=3)` of orthonormal
      material frames at the nodes.
    fiber_inner_radii: Array of shape `(num_elems, num_fibers)` [m] of each
      sector's inner radius. Use 0 for a solid disk.
    fiber_outer_radii: Array of shape `(num_elems, num_fibers)` [m] of each
      sector's outer radius.
    fiber_start_angles: Array of shape `(num_elems, num_fibers)` [rad] of each
      sector's start angle, measured in the cross-section plane from `d1` towards
      `d2`.
    fiber_end_angles: Array of shape `(num_elems, num_fibers)` [rad] of each
      sector's end angle. A span of `2π` gives a full ring.
    fiber_center_offsets: Optional array of shape `(num_elems, num_fibers,
      num_crossec_dim=2)` [m] of each sector's centre in the `d1`-`d2` plane,
      following the `Fibers.offsets` convention. If None, all sectors are centred
      on the rod centreline.
    num_arc_segments: Number of flat quads used to approximate each arc. Increase
      for wide spans (e.g. 24+ for a full ring).
    fiber_field: Optional array of shape `(num_elems, num_fibers)` of scalars used
      to colour each cell through `cmap`, or `(num_elems, num_fibers, 3 | 4)` of
      RGB(A) colours used as they are. If None, a flat facecolor is used.
    fiber_field_val_range: Optional `(min, max)` colour-map limits. If None, the
      local minimum and maximum are used.
    fiber_mask: Optional boolean array of shape `(num_elems, num_fibers)`. Cells
      where it is False are not drawn at all.
    cmap: Matplotlib colormap name or colormap instance.
    alpha: Face opacity in `(0, 1]`.
    edge_color: Matplotlib colour for the cell outlines.
    edge_width: Line width of the cell outlines. Set to 0 to hide them.
    plot_colorbar: Attach a colorbar when a field is given.
    colorbar_label: Label for this field's colorbar.
    colorbar_pad: Padding between the axis and the colorbar.
    ax: Optional pre-existing Matplotlib 3D axis. If None, one is created.

  Returns:
    ax: The Matplotlib 3D Axes, with the viewport re-framed around the bundle.
  """
  ax = _prepare_axes(ax)
  cells = _fiber_annular_sector_prisms(
    node_coords=np.asarray(node_coords),
    directors=np.asarray(directors),
    fiber_inner_radii=np.asarray(fiber_inner_radii),
    fiber_outer_radii=np.asarray(fiber_outer_radii),
    fiber_start_angles=np.asarray(fiber_start_angles),
    fiber_end_angles=np.asarray(fiber_end_angles),
    fiber_center_offsets=(
      None if fiber_center_offsets is None else np.asarray(fiber_center_offsets)
    ),
    num_arc_segments=num_arc_segments,
  )
  quads, values = _flatten_cells(cells, fiber_field, fiber_mask)
  if quads.shape[0] == 0:
    return ax
  _add_patches(
    ax=ax,
    quads=quads,
    field=values,
    field_val_range=fiber_field_val_range,
    cmap=cmap,
    alpha=alpha,
    edge_color=edge_color,
    edge_width=edge_width,
    plot_colorbar=plot_colorbar,
    colorbar_label=colorbar_label,
    colorbar_pad=colorbar_pad,
  )
  _extend_view(ax, quads)
  return ax


gel_colors = {
  "water": np.array([200, 245, 255]) / 255,  # #C8F5FF
  "hydrogel": np.array([103, 158, 248]) / 255,  # #679EF8
  "white": np.array([243, 246, 248]) / 255,  # #F3F6F8
  "elastomer": np.array([226, 119, 219]) / 255,  # #E277DB
}

mag_colors = {
  "mag_north": np.array([16, 67, 255]) / 255,  # #1043FF
  "white": np.array([240, 244, 248]) / 255,  # #F0F4F8
  "mag_south": np.array([255, 30, 67]) / 255,  # #FF1E43
}


elast_gel_cmap = LinearSegmentedColormap.from_list(
  "elast_gel_cmap",
  np.vstack([gel_colors["elastomer"], gel_colors["white"], gel_colors["hydrogel"]]),
)

# Diverging south -> unmagnetised -> north, for a signed magnetisation component.
mag_cmap = LinearSegmentedColormap.from_list(
  "mag_cmap",
  np.vstack([mag_colors["mag_south"], mag_colors["white"], mag_colors["mag_north"]]),
)

# Sequential yellow -> dark red for a contact force magnitude, with the near-white start
# of YlOrRd cut off so weak contacts stay visible on a white background.
contact_force_cmap = LinearSegmentedColormap.from_list(
  "contact_force_cmap", plt.colormaps["YlOrRd"](np.linspace(0.3, 1.0, 256))
)

high_res_plot_settings = {
  "figure.dpi": 350,
  "savefig.dpi": 350,
  "font.family": "serif",
  "font.size": 24,
  "axes.labelsize": 18,
  "axes.titlesize": 18,
  "xtick.labelsize": 18,
  "ytick.labelsize": 18,
  "legend.fontsize": 18,
  "text.usetex": False,
  "lines.linewidth": 1.5,
  "axes.linewidth": 1.0,
  "grid.linestyle": ":",
  "grid.linewidth": 0.5,
  "savefig.bbox": "tight",
  "savefig.format": "pdf",
}

# default plot settings for everyday plots
default_plot_settings = {
  "figure.dpi": 100,
  "savefig.dpi": 100,
  "font.family": "serif",
  "font.size": 10,
  "axes.labelsize": 12,
  "axes.titlesize": 14,
  "xtick.labelsize": 10,
  "ytick.labelsize": 10,
  "legend.fontsize": 10,
  "text.usetex": False,
  "lines.linewidth": 1.0,
  "axes.linewidth": 0.8,
  "grid.linestyle": "--",
  "grid.linewidth": 0.5,
  "savefig.bbox": "tight",
  "savefig.format": "pdf",
}
