"""Boundary condition for cosserat rods."""

import abc
import dataclasses
import enum
from typing import List, Tuple, Union

import jax.experimental.sparse as jax_sprs
import jax.numpy as jnp
import numpy as np

import crops.src.mesher as _mesher
import crops.src.utils as _utils


class BCType(enum.Enum):
  """Types of boundary conditions supported."""

  DIRICHLET = enum.auto()
  NEUMANN = enum.auto()


class BC(abc.ABC):
  """Abstract base class for boundary conditions.

  Subclasses implement `process_step` to extract global dof indices and associated bc
  values. Supports load / time stepping.

  Attributes:
    type: `BCType` tag identifying the type of boundary condition.
    name: Optional label for the boundary condition.
  """

  def __init__(self, bc_type: BCType, name: str = "") -> None:
    """Initializes the boundary condition."""
    self.type = bc_type
    self.name = name

  @abc.abstractmethod
  def process_step(
    self, rod: _mesher.Rod, step_idx: int = 0
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extracts global degree-of-freedom indices and associated values."""
    pass


class DirichletBC(BC):
  """Prescribed displacement or rotation constraints on selected rod nodes.

  Enforces u = u₀ on the Dirichlet boundary 𝛤_D.
  """

  def __init__(
    self,
    values: Tuple[_utils.Translate_or_Rotate, List[np.ndarray]],
    node_targets: List[np.ndarray],
    name: str = "",
  ) -> None:
    """Initializes the Dirichlet constraint schedule.

    Attributes:
      values: Tuple `(dof_enum, val_arrays)` where `dof_enum` is a `_utils.Translation`
        or `_utils.Rotation` member identifying the  constrained component, and
        `val_arrays` is a list of length `num_steps` of arrays of shape
        `(num_target_nodes_k,)`. The k-th array contains the prescribed values for the
        k-th step. The array sizes can vary across steps.
      node_targets: List of length `num_steps` of 1D integer arrays, each of shape
        `(num_target_nodes_k,)`. Nodes with prescribed translational or rotational
        dofs. The array sizes can vary across steps.
      name: Optional label for this condition.
    """

    super().__init__(BCType.DIRICHLET, name)
    self.values = values
    self.node_targets = node_targets

  def process_step(
    self, rod: _mesher.Rod, step_idx: int = 0
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extracts constrained global dof indices and prescribed values at one step.

    Args:
      rod: `Rod` instance of the rod to which the boundary condition is applied.
      step_idx: Step index into the schedule.

    Returns:
      Tuple of:
        dirichlet_dofs: Integer array of shape `(num_constrained_dofs,)`.
        dirichlet_values: Float array of shape `(num_constrained_dofs,)`.
    """
    dof_enum, val_arrays = self.values
    dof_indices = rod.get_dof_indices(
      dof_enum=dof_enum, node_indices=self.node_targets[step_idx]
    )
    return dof_indices, jnp.asarray(val_arrays[step_idx])


class NeumannBC(BC):
  """Prescribed external force or concentrated moment on selected rod nodes.

  Attributes:
    values: Tuple `(dof_enum, val_arrays)` where `dof_enum` is a `_utils.Translation` or
      `_utils.Rotation` member identifying the loaded component, and `val_arrays` is a
      list of length `num_steps` of float arrays of shape `(num_target_nodes_k,)`. The
      array sizes can vary across steps.
    node_targets: List of length `num_steps` of 1D integer arrays, each of shape
      `(num_target_nodes_k,)`. Nodes receiving point forces (translation) or moments
      (rotation).
    name: Optional label for this condition.
  """

  def __init__(
    self,
    values: Tuple[_utils.Translate_or_Rotate, List[np.ndarray]],
    node_targets: List[np.ndarray],
    name: str = "",
  ) -> None:
    """Initializes the Neumann load schedule."""
    super().__init__(BCType.NEUMANN, name)
    self.values = values
    self.node_targets = node_targets

  def process_step(
    self, rod: _mesher.Rod, step_idx: int = 0
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Extracts loaded global dof indices and force magnitudes at one step.

    Args:
      rod: `Rod` instance to which the boundary condition is applied.
      step_idx: Step index into the schedule.

    Returns:
      Tuple of:
        load_dofs: Integer array of shape `(num_loaded_dofs,)`.
        load_values: Float array of shape `(num_loaded_dofs,)`.
    """
    dof_enum, val_arrays = self.values
    val_k = val_arrays[step_idx]
    dof_indices = rod.get_dof_indices(
      dof_enum=dof_enum, node_indices=self.node_targets[step_idx]
    )

    return dof_indices, jnp.asarray(val_k)


@dataclasses.dataclass
class BoundaryConditions:
  """Preprocessed boundary condition schedule for a single rod.

  Stores all boundary data as JAX arrays with a leading `num_steps` axis. At each step
  k, `dirichlet_mask[k]` identifies which dofs are prescribed, `dirichlet_values[k]`
  holds the corresponding prescribed values (zero at unconstrained dofs), and
  `external_forces[k]` holds the assembled Neumann loads.

  Attributes:
    dirichlet_mask: Boolean array of shape `(num_steps, num_dofs)`. Entry `[k, i]` is
      `True` when DOF i is prescribed at step k.
    dirichlet_values: Float array of shape `(num_steps, num_dofs)` [m or rad].
      Prescribed DOF values at step k. Default is zero at unconstrained dofs.
    external_forces: Float array of shape `(num_steps, num_dofs)` [N or N·m].
      Assembled Neumann point forces and moments at step k.
  """

  dirichlet_mask: jnp.ndarray
  dirichlet_values: jnp.ndarray
  external_forces: jnp.ndarray

  @property
  def num_steps(self) -> int:
    """Number of steps in the schedule."""
    return int(self.dirichlet_mask.shape[0])

  def free_dofs(self, step_idx: int = 0) -> jnp.ndarray:
    """Returns a boolean mask of shape `(num_dofs,)` where `True` marks unconstrained
    degrees of freedom at step `step_idx`.

    Args:
      step_idx: Step index. Defaults to 0.

    Returns:
      Boolean array of shape `(num_dofs,)` indicating unconstrained dofs.
    """
    return jnp.logical_not(self.dirichlet_mask[step_idx])


def process_boundary_conditions(
  rod: _mesher.Rod,
  boundary_conditions_list: List[Union[DirichletBC, NeumannBC]],
) -> BoundaryConditions:
  """Assembles a `BoundaryConditions` schedule from a list of `BC` instances.

  NOTE: We infer `num_steps` from the `values` in the first `BC` instance. All instances
  must therefore have the same number of steps. Each step is assembled by calling
  `process_step` on every instance and writing results into arrays of `(num_dofs,)`,
  which are then stacked into the final `(num_steps, num_dofs)` schedule arrays.

  Args:
    rod: `Rod` instance of the rod for which to process boundary conditions.
    boundary_conditions_list: List of `DirichletBC` and `NeumannBC` instances the
      target dofs and the corresponding values.

  Returns:
    bcs: `BoundaryConditions` data structure with the processed boundary conditions.

  Raises:
    ValueError: When `num_steps` cannot be inferred or is inconsistent across instances.
  """
  num_steps = None
  for bc_item in boundary_conditions_list:
    _, val_arrays = bc_item.values
    if num_steps is None:
      num_steps = len(val_arrays)
    elif len(val_arrays) != num_steps:
      raise ValueError(
        f"Inconsistent num_steps: expected {num_steps}, "
        f"got {len(val_arrays)} in {bc_item.name}."
      )
  if num_steps is None:
    raise ValueError("No BC instances found in boundary_conditions_list.")

  mask_steps = []
  vals_steps = []
  fext_steps = []

  for step_idx in range(num_steps):
    step_dirichlet_mask = np.zeros(rod.num_dofs, dtype=bool)
    step_dirichlet_values = np.zeros(rod.num_dofs)
    step_external_forces = np.zeros(rod.num_dofs)

    for bc_item in boundary_conditions_list:
      dofs_k, data_k = bc_item.process_step(rod=rod, step_idx=step_idx)
      if isinstance(bc_item, NeumannBC):
        step_external_forces[dofs_k] += data_k
      elif isinstance(bc_item, DirichletBC):
        step_dirichlet_mask[dofs_k] = True
        step_dirichlet_values[dofs_k] = data_k

    mask_steps.append(step_dirichlet_mask)
    vals_steps.append(step_dirichlet_values)
    fext_steps.append(step_external_forces)

  return BoundaryConditions(
    dirichlet_mask=jnp.asarray(np.stack(mask_steps)),
    dirichlet_values=jnp.asarray(np.stack(vals_steps)),
    external_forces=jnp.asarray(np.stack(fext_steps)),
  )


def apply_dirichlet_bc(
  jacobian: jax_sprs.BCOO,
  dirichlet_mask: jnp.ndarray,
) -> jax_sprs.BCOO:
  """Enforces Dirichlet constraints on the given tangent matrix (jacobian).

  Zeros all entries whose row or column index corresponds to a constrained DOF, then
  appends diagonal entries of 1.0 at constrained dofs and 0.0 elsewhere to restore
  non-singularity.

  Args:
    jacobian: Sparse tangent matrix in JAX BCOO format, shape `(num_dofs, num_dofs)`.

    dirichlet_mask: Bool array of shape `(num_dofs,)`. `True` at each constrained dof.

  Returns:
    constrained_jacobian: BCOO matrix with Dirichlet rows and columns zeroed and
      identity entries on the constrained diagonal.
  """
  data, indices = jacobian.data, jacobian.indices

  row_constrained = dirichlet_mask[indices[:, 0]]
  col_constrained = dirichlet_mask[indices[:, 1]]
  either_constrained = jnp.logical_or(row_constrained, col_constrained)
  data = jnp.where(either_constrained, 0.0, data)

  num_dofs = dirichlet_mask.shape[0]
  diag_idx = jnp.stack([jnp.arange(num_dofs), jnp.arange(num_dofs)], axis=1)
  diag_data = dirichlet_mask.astype(data.dtype)

  return jax_sprs.BCOO(
    (jnp.concatenate([data, diag_data]), jnp.concatenate([indices, diag_idx])),
    shape=jacobian.shape,
  )
