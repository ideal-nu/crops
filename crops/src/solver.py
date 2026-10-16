"""Linear and Nonlinear solver for the system of equations integrated into JAX."""

import enum
import functools
import warnings
from typing import Any, Callable, Dict, TypeAlias, Union

import jax
import jax.experimental.sparse as jax_sprs
import jax.numpy as jnp
import jax_dataclasses as jxd
import numpy as np
import scipy.sparse as spy_sprs
import scipy.sparse.linalg as spy_linalg
from jax.typing import ArrayLike

try:
  import petsc4py.PETSc as PETSc
  import pyamg
  import pypardiso  # type: ignore
  import scikits.umfpack as umf
except ImportError:
  warnings.warn("Some solvers may not be available.")

Jacobian: TypeAlias = Union[jax_sprs.BCOO, jnp.ndarray]


class LinearSolvers(enum.Enum):
  """Linear solvers wrapped around with `jax.lax.custom_linear_solve`."""

  LINALG_SOLVE = enum.auto()
  AMG_CG = enum.auto()
  AMG_BICGSTAB = enum.auto()
  SCIPY_SPARSE = enum.auto()
  PARDISO = enum.auto()
  UMFPACK = enum.auto()
  PETSC = enum.auto()


class NonlinearSolvers(enum.Enum):
  """Enumeration of nonlinear solvers.
  
  Attributes:
    NEWTON_RAPHSON: Standard Newton-Raphson method.
    MODIFIED_NEWTON_RAPHSON: Modified Newton-Raphson method with half-step damping.
    ARMIJO_BACKTRACKING_NEWTON_RAPHSON: Newton-Raphson method with Armijo backtracking 
      line search.
    UNROLLED_NEWTON_RAPHSON: Unrolled Newton-Raphson method for debugging purposes.
  """

  NEWTON_RAPHSON = enum.auto()
  MODIFIED_NEWTON_RAPHSON = enum.auto()
  ARMIJO_BACKTRACKING_NEWTON_RAPHSON = enum.auto()
  UNROLLED_NEWTON_RAPHSON = enum.auto()  # For debugging only


@jxd.pytree_dataclass
class NRStats:
  """Convergence statistics of one Newton-Raphson solve.

  All fields are scalar arrays so that the stats can flow through `jax.jit`,
    `jax.lax.scan` (where they get stacked along a leading axis, one entry per load
    step) and the custom JVP rules of the kernels (with zero tangents).

  Attributes:
    num_iters: Number of Newton iterations performed.
    res_norm: Residual norm tested against the convergence criterion at termination.
      For `ARMIJO_BACKTRACKING_NEWTON_RAPHSON` this is the residual of the returned
      iterate. `NEWTON_RAPHSON` and `MODIFIED_NEWTON_RAPHSON` update the iterate once
      more after evaluating this residual, so there it is the residual of the iterate
      *before* the last update.
    init_res_norm: Residual norm at the initial guess.
    converged: `True` if `res_norm` satisfied the convergence criterion, `False` if
      the solver stopped because `max_iter` was reached.
  """

  num_iters: jnp.ndarray
  res_norm: jnp.ndarray
  init_res_norm: jnp.ndarray
  converged: jnp.ndarray


def _jacobi_preconditioner(A: spy_sprs.coo_matrix) -> spy_sprs.coo_matrix:
  """Computes the Jacobi preconditioner for a sparse matrix A in COO format.

  Args:
    A: The sparse matrix in COO format.

  Returns: The Jacobi preconditioner in COO format.
  """

  # Extract diagonal elements
  diag_data = A.diagonal()

  # Ensure no zeros on the diagonal (avoid division by zero)
  diag_data[diag_data == 0.0] = 1.0

  diag_idxs = np.arange(A.shape[0])
  # Construct the preconditioner as a diagonal matrix
  M_inv = spy_sprs.coo_matrix((1.0 / diag_data, (diag_idxs, diag_idxs)), shape=A.shape)
  return M_inv


def _petsc_solve(
  A: spy_sprs.csr_matrix, b: ArrayLike, solver_options: Dict
) -> np.ndarray:
  """Solve for u = A^{-1}b using PETSc.

  Args:
    A: A sparse CSR matrix of shape (m,m).
    b: The rhs vector of shape (m,).
    solver_options: A dictionary containing PETSc solver options.

  Returns: The solution vector of shape (m,).
  """
  ksp_type = (
    solver_options["petsc_solver"]["ksp_type"]
    if "ksp_type" in solver_options["petsc_solver"]
    else "bcgsl"
  )
  pc_type = (
    solver_options["petsc_solver"]["pc_type"]
    if "pc_type" in solver_options["petsc_solver"]
    else "ilu"
  )

  A = PETSc.Mat().createAIJ(
    size=A.shape,
    csr=(
      A.indptr.astype(PETSc.IntType, copy=False),
      A.indices.astype(PETSc.IntType, copy=False),
      A.data,
    ),
  )

  rhs = PETSc.Vec().createSeq(len(b))
  rhs.setValues(range(len(b)), np.array(b))
  ksp = PETSc.KSP().create()
  ksp.setOperators(A)
  ksp.setFromOptions()
  ksp.setType(ksp_type)
  ksp.pc.setType(pc_type)

  if ksp_type == "tfqmr":
    ksp.pc.setFactorSolverType("mumps")

  x = PETSc.Vec().createSeq(len(b))
  ksp.solve(rhs, x)

  return x.getArray()


def _safe_divide(x: jax.Array, y: jax.Array, eps: float = 1.0e-8) -> jax.Array:
  """Compute the division of x by y with a safe check for division by zero.

  This function ensures that the denominator `y` is non-zero before applying the
    division operation. If `y` is zero, it returns zero. This ensures that the
    division operation does not result in undefined behavior.

  Args:
    x: Numerator array.
    y: Denominator array.
    eps: Small value below which the absolute value of the denominator is
      treated as zero.

  Returns: The result of `x / y` if `y` is non-zero, otherwise zero.
  """
  mask = jnp.abs(y) < eps
  z = jnp.where(mask, 1.0, y)
  return jnp.where(mask, 0.0, x / z)


def linear_solve(
  A: Jacobian,
  b: jnp.ndarray,
  params: Dict,
  u0: jnp.ndarray = None,
) -> jnp.ndarray:
  """Solve for u = A^{-1}b using custom solvers.

  This function uses `jax.lax.custom_linear_solve` to solve the linear system of
    equations defined by the sparse matrix A and the right-hand side vector b. For
    the list of available solvers, see `LinearSolvers` enum. The solvers themselves
    are not differentiable. By wrapping them in `jax.lax.custom_linear_solve`, we can
    use them in JAX computations while still allowing for differentiation through the
    linear solve operation.

  Args:
    A: A sparse BCOO matrix of shape (m,m) or a dense array of shape (m,m) representing
       the linear operator. In the case of A being dense, the function will use
       `jnp.linalg.solve` to solve the system. If A is sparse, the function will use
       the specified solver in `params` to solve the system.
    b: The rhs vector of shape (m,).
    params: Additional parameters for the solver. This should include:
      - "solver": The type of solver to use, as defined in `LinearSolvers`
      - other solver-specific parameters, e.g., "rtol" for relative tolerance.
    u0: The initial guess for the solution of shape (m,). Not required for all solvers,
      but some solvers may need it (e.g., iterative solvers like CG or BICGSTAB).

  Returns: The solution vector of shape (m,).
  """
  # If A is a scalar, we simply return b/A
  if isinstance(A, (float, int)) or (
    isinstance(A, (jnp.ndarray, np.ndarray)) and A.ndim == 0
  ):
    return _safe_divide(b, A)

  # If A is dense, we ignore the requested sparse solver and use standard linalg
  if isinstance(A, jnp.ndarray):
    return jnp.linalg.solve(A, b)

  def mv(u):
    return A @ u

  def solver_wrapper(A, b):
    A_sp = spy_sprs.coo_matrix(
      (jax.lax.stop_gradient(A.data), (A.indices[:, 0], A.indices[:, 1])), shape=A.shape
    )
    b = jax.lax.stop_gradient(b)

    if params["solver"] == LinearSolvers.AMG_CG:
      M = _jacobi_preconditioner(A_sp)
      x, _ = pyamg.krylov.cg(A_sp, b, tol=params["rtol"], x0=u0, M=M)

    elif params["solver"] == LinearSolvers.AMG_BICGSTAB:
      if A_sp.diagonal().sum() < 0:
        A_sp = -A_sp
        b = -b
      A_csr = A_sp.tocsr()
      ml = pyamg.smoothed_aggregation_solver(A_csr)
      M = ml.aspreconditioner()
      x, _ = spy_linalg.bicgstab(A_csr, b, rtol=params["rtol"], x0=u0, M=M)

    elif params["solver"] == LinearSolvers.SCIPY_SPARSE:
      x = spy_linalg.spsolve(A_sp.tocsr(), b)

    elif params["solver"] == LinearSolvers.LINALG_SOLVE:
      x = jnp.linalg.solve(A, b)

    elif params["solver"] == LinearSolvers.PARDISO:
      x = pypardiso.spsolve(A_sp.tocsr(), np.asarray(b))

    elif params["solver"] == LinearSolvers.UMFPACK:
      x = umf.spsolve(A_sp.tocsr(), np.asarray(b))

    elif params["solver"] == LinearSolvers.PETSC:
      x = _petsc_solve(A_sp.tocsr(), np.asarray(b), params)

    else:
      raise ValueError("Invalid solver type")

    return x.astype(b.dtype).reshape(b.shape)

  result_shape = jax.ShapeDtypeStruct(b.shape, b.dtype)

  def cust_fwd_solver(mv, b):
    return jax.pure_callback(solver_wrapper, result_shape, A, b)

  def cust_bwd_solver(mv, b):
    return jax.pure_callback(solver_wrapper, result_shape, A.T, b)

  sol = jax.lax.custom_linear_solve(
    mv, b, solve=cust_fwd_solver, transpose_solve=cust_bwd_solver
  )
  return sol.reshape(-1)


def nonlinear_problem(func: Callable):
  """Decorator to wrap a simple residual function f(x, *params).

  Args:
    func: A function that takes in (x, *params) and returns the residual of
      shape (n,).

  Returns a function that returns (residual, dense_jacobian). The dense jacobian is of
    shape (n, n) and is computed using `jax.jacobian`. This is used in the nonlinear
    solvers to compute the tangent stiffness matrix. Note that the jacobian is dense.
    For large problems, this may not be efficient, and we may want to implement a
    custom jacobian function that returns a sparse matrix.
  """

  def wrapper(x, *params):
    res = func(x, *params)
    jac = jax.jacobian(func, argnums=0)(x, *params)
    return res, jac

  return wrapper


def _convergence_target(nl_settings: Dict, init_res_norm: jnp.ndarray) -> jnp.ndarray:
  """Residual norm at or below which the Newton iteration is considered converged."""
  rel_target = nl_settings["threshold"] * init_res_norm
  abs_target = nl_settings.get("abs_threshold", 1e-10)
  return jnp.maximum(rel_target, abs_target)


def _nr_stats(
  nl_settings: Dict,
  num_iters: jnp.ndarray,
  res_norm: jnp.ndarray,
  init_res_norm: jnp.ndarray,
) -> NRStats:
  """Bundles the final state of a Newton loop into an `NRStats`."""
  return NRStats(
    num_iters=jnp.asarray(num_iters, dtype=jnp.int32),
    res_norm=res_norm,
    init_res_norm=init_res_norm,
    converged=res_norm <= _convergence_target(nl_settings, init_res_norm),
  )


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 2))
def _standard_nr_kernel(
  get_res_and_tgt_fn: Callable,
  x0: jnp.ndarray,
  settings: Dict,
  *params,
) -> tuple[jnp.ndarray, NRStats]:
  """Nonlinear solver using Newton's method.

  This function implements the Newton-Raphson method to solve the nonlinear system of
    equations as defined by the `get_res_and_tgt_fn` function. The method iteratively
    updates the  solution until convergence is achieved based on the specified settings
    in `settings`.

    We then define a custom JVP rule for this function to enable differentiation
    through the nonlinear solve operation. In particular, we use the implicit function
    theorem to compute the JVP of the solution with respect to the parameters.
    This avoids the need to differentiate through the entire Newton-Raphson iteration
    chain, which can be computationally expensive and memory-intensive.

  Args:
    get_res_and_tgt_fn: Function that computes the residual and tangents.
    jacobian_fn: The function to compute the jacobian. The function takes the
      same arguments as residual_fn.
    x0: Array of (n,) of the initial guess.
    settings: Additional parameters for the linear and nonlinear solvers.
    *params: Additional parameters for the residual and jacobian functions.

  Returns: A tuple of the array of (n,) of the solution and the `NRStats` of the solve.
  """
  ctr = 0
  res, _ = get_res_and_tgt_fn(x0, False, *params)
  init_res_norm = jax.lax.stop_gradient(jnp.linalg.norm(res))

  state = (x0, init_res_norm, ctr)
  nl_settings = settings["nonlinear"]
  lin_settings = settings["linear"]

  def cond_fn(state):
    _, res_norm, ctr = state
    cond_iter = ctr < nl_settings["max_iter"]
    cond_res = res_norm > _convergence_target(nl_settings, init_res_norm)
    return cond_iter & cond_res

  def body_fn(state):
    x0, res_norm, ctr = state
    residual, jacobian = get_res_and_tgt_fn(x0, True, *params)

    x0 -= linear_solve(jacobian, residual, params=lin_settings)
    res_norm = jax.lax.stop_gradient(jnp.linalg.norm(residual))
    ctr += 1

    return (x0, res_norm, ctr)

  x0, res_norm, ctr = jax.lax.while_loop(cond_fn, body_fn, state)
  return x0, _nr_stats(nl_settings, ctr, res_norm, init_res_norm)


@_standard_nr_kernel.defjvp
def solve_jvp(
  get_res_and_tgt_fn: Callable,
  settings: Dict,
  primals: tuple[jnp.ndarray, ...],
  tangents: tuple[jnp.ndarray, ...],
) -> tuple[tuple[jnp.ndarray, NRStats], tuple[jnp.ndarray, NRStats]]:
  """JVP rule for the Newton-Raphson solver.

  This function implements the JVP rule for the Newton-Raphson solver using the
    implicit function theorem. Given the primal inputs (x0, *params) and their tangents,
    we first solve the nonlinear system to obtain the solution x. We then compute the
    product of the derivative of the residual with respect to the parameters times
    the perturbation of the parameters. Finally, we solve the linear system to
    obtain the JVP of the solution with respect to the parameters. The `NRStats` are
    not differentiable and get zero tangents.

  Args:
    problem: The nonlinear problem object.
    primals: A tuple containing the primal inputs (x0, *params).
    tangents: A tuple containing the tangents of the primal inputs (dx0, *dparams).

  Returns: A tuple containing the primal outputs (x, stats) and their tangents
    (dx, zero stats).
  """
  x0, *params = primals
  _, *dparams = tangents

  x_star, stats = _standard_nr_kernel(get_res_and_tgt_fn, x0, settings, *params)

  _, jacobian = get_res_and_tgt_fn(x_star, True, *params)

  def residual_only(x, *p):
    r, _ = get_res_and_tgt_fn(x, False, *p)
    return r

  _, rhs_vec = jax.jvp(lambda *p: residual_only(x_star, *p), params, dparams)

  dx = linear_solve(jacobian, -rhs_vec, params=settings["linear"])

  dstats = jax.tree.map(jax.custom_derivatives.zero_from_primal, stats)
  return (x_star, stats), (dx, dstats)


def _unrolled_nr_kernel(
  get_res_and_tgt_fn: Callable,
  x0: jnp.ndarray,
  settings: Dict,
  *params,
) -> tuple[jnp.ndarray, NRStats]:
  """Unrolled Nonlinear solver using Newton's method for debugging.

  This function implements the Newton-Raphson method using a python loop
  instead of jax.lax.while_loop. This allows for easier debugging and
  inspection of intermediate values, but unrolls the loop during JIT compilation.

  Args:
    get_res_and_tgt_fn: Function that computes the residual and tangents.
    x0: Array of (n,) of the initial guess.
    settings: Additional parameters for the linear and nonlinear solvers.
    *params: Additional parameters for the residual and jacobian functions.

  Returns: A tuple of the array of (n,) of the solution and the `NRStats` of the solve.
  """
  nl_settings = settings["nonlinear"]
  lin_settings = settings["linear"]

  res, _ = get_res_and_tgt_fn(x0, False, *params)
  init_res_norm = jnp.linalg.norm(res)
  target = _convergence_target(nl_settings, init_res_norm)

  res_norm, num_iters = init_res_norm, 0
  for _ in range(nl_settings["max_iter"]):
    residual, jacobian = get_res_and_tgt_fn(x0, True, *params)
    res_norm = jnp.linalg.norm(residual)

    converged = res_norm <= target
    delta = linear_solve(jacobian, residual, params=lin_settings)

    x0 = jnp.where(converged, x0, x0 - delta)
    num_iters = num_iters + (~converged).astype(jnp.int32)

  return x0, _nr_stats(nl_settings, num_iters, res_norm, init_res_norm)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 2))
def _modified_nr_kernel(
  get_res_and_tgt_fn: Callable,
  x0: jnp.ndarray,
  settings: Dict,
  *params,
) -> tuple[jnp.ndarray, NRStats]:
  """Modified Newton-Raphson solver for nonlinear problems.

  This function implements a modified version of the Newton-Raphson method that
    uses a half-step approach to improve convergence. The method iteratively updates the
    solution until convergence is achieved based on the specified settings in
    `settings`.

  Args:
    get_res_and_tgt_fn: Function that computes the residual and tangents.
    x0: Initial guess for the solution, an array of shape (num_dofs,).
    settings: Additional parameters for the linear and nonlinear solvers.
    *params: Additional parameters required by the problem's residual and tangent
      stiffness functions.

  Returns: A tuple of the solution vector of shape (num_dofs,) and the `NRStats` of
    the solve.
  """
  ctr = 0
  init_res, _ = get_res_and_tgt_fn(x0, False, *params)
  init_res_norm = jax.lax.stop_gradient(jnp.linalg.norm(init_res))

  state = (x0, init_res_norm, ctr)
  nl_settings = settings["nonlinear"]
  lin_settings = settings["linear"]

  def cond_fn(state):
    _, res_norm, ctr = state
    cond_iter = ctr < nl_settings["max_iter"]
    cond_res = res_norm > _convergence_target(nl_settings, init_res_norm)
    return cond_iter & cond_res

  def body_fn(state):
    x0, _, ctr = state
    # current step
    residual, jacobian = get_res_and_tgt_fn(x0, True, *params)

    res_norm_curr = jax.lax.stop_gradient(jnp.linalg.norm(residual))
    delta_x = linear_solve(jacobian, residual, params=lin_settings)

    # half step
    x0_half_step = x0 - 0.5 * delta_x
    residual_half_step, _ = get_res_and_tgt_fn(x0_half_step, False, *params)
    res_norm_half_step = jax.lax.stop_gradient(jnp.linalg.norm(residual_half_step))

    # full step
    x0_full_step = x0 - delta_x
    residual_full_step, _ = get_res_and_tgt_fn(x0_full_step, False, *params)
    res_norm_full_step = jax.lax.stop_gradient(jnp.linalg.norm(residual_full_step))

    numer = 3.0 * res_norm_curr + res_norm_full_step - 4.0 * res_norm_half_step
    denom = 4.0 * res_norm_curr + 4.0 * res_norm_full_step - 8.0 * res_norm_half_step
    lam = jnp.where(jnp.abs(denom) > 1e-30, numer / denom, 1.0)
    lam = jnp.where(jnp.isfinite(lam), lam, 1.0)
    lam = jnp.clip(lam, min=nl_settings.get("lam_min", 0.01), max=1.0)

    x0 = x0 - lam * delta_x

    ctr += 1
    return (x0, res_norm_curr, ctr)

  x0, res_norm, ctr = jax.lax.while_loop(cond_fn, body_fn, state)
  return x0, _nr_stats(nl_settings, ctr, res_norm, init_res_norm)


@_modified_nr_kernel.defjvp
def solve_jvp(  # noqa: F811
  get_res_and_tgt_fn: Callable,
  settings: Dict,
  primals: tuple[jnp.ndarray, ...],
  tangents: tuple[jnp.ndarray, ...],
) -> tuple[tuple[jnp.ndarray, NRStats], tuple[jnp.ndarray, NRStats]]:
  """JVP rule for the Newton-Raphson solver.

  This function implements the JVP rule for the Newton-Raphson solver using the
    implicit function theorem. Given the primal inputs (x0, *params) and their tangents,
    we first solve the nonlinear system to obtain the solution x. We then compute the
    product of the derivative of the residual with respect to the parameters times
    the perturbation of the parameters. Finally, we solve the linear system to
    obtain the JVP of the solution with respect to the parameters. The `NRStats` are
    not differentiable and get zero tangents.

  Args:
    get_res_and_tgt_fn: The function to compute the residual and tangents. The function
      takes the same arguments as the primal inputs (x0, *params).
    settings: Additional parameters for the linear and nonlinear solvers.
    primals: A tuple containing the primal inputs (x0, *params).
    tangents: A tuple containing the tangents of the primal inputs (dx0, *dparams).

  Returns: A tuple containing the primal outputs (x, stats) and their tangents
    (dx, zero stats).
  """
  x0, *params = primals
  _, *dparams = tangents

  x_star, stats = _modified_nr_kernel(get_res_and_tgt_fn, x0, settings, *params)

  _, jacobian = get_res_and_tgt_fn(x_star, True, *params)

  def residual_only(x, *p):
    r, _ = get_res_and_tgt_fn(x, False, *p)
    return r

  _, rhs_vec = jax.jvp(lambda *p: residual_only(x_star, *p), params, dparams)
  dx = linear_solve(jacobian, -rhs_vec, params=settings["linear"])

  dstats = jax.tree.map(jax.custom_derivatives.zero_from_primal, stats)
  return (x_star, stats), (dx, dstats)


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 2))
def _damped_armijo_nr_kernel(
  get_res_and_tgt_fn: Callable,
  x0: jnp.ndarray,
  settings: Dict,
  *params,
  eps: float = 1e-30,
) -> tuple[jnp.ndarray, NRStats]:
  """Solve a nonlinear system with Newton damping + Armijo backtracking.

  We solve the nonlinear residual equation
      r(x) = 0,
  using Newton iteration. At iteration k we compute a Newton direction by solving
      J(x_k) Δx_k = r(x_k),
  and propose an update
      x_{k+1} = x_k - α_k Δx_k.

  Two robustness mechanisms are used:

  1) Modified-Newton damping predictor:
    A scalar step-length predictor `lam_pred ∈ [lam_min, 1]` is estimated using res
    norms at a half-step and full-step:
        x_half = x - 0.5 Δx,
        x_full = x - 1.0 Δx,
    and then computing `lam_pred` from these three norms. This provides a cheap, prob-
    dependent initial guess for the step length.

  2) Backtracking (Armijo) line search:
   Starting from `α = lam_pred`, we shrink `α ← α * line_search_shrink` until Armijo
   sufficient decrease condition is satisfied:
       ||r(x - αΔx)|| <= (1 - c α) ||r(x)||,
   or until α drops below `line_search_alpha_min`. If no acceptable α is found, the
   iteration rejects the step and keeps `x` unchanged for that iteration.

  Nonlinear settings (required) in `problem.solver_settings["nonlinear"]`:

    - max_iter (int, >= 1):Maximum Newton iterations per solve.

    - threshold (float, (0, 1)):
        Relative convergence tolerance on residual norm: stop when
          ||r|| <= threshold * ||r0||.
        Typical values 1e-8 to 1e-12. Smaller is stricter (more iterations).

    - lam_min (float, (0, 1]):
        Lower bound for the predicted step length (lam_pred) before line search.
        Purpose: avoid tiny predictor steps caused by noisy half/full norms.
        Typical values 0.01–0.2. If too small, Newton can stagnate; if too large, the
        line search will do more work (or reject more often).

    - line_search_max_iter (int, >= 0):
        Max backtracking reductions. Typical values 8–20.
        More iterations increases robustness at the cost of extra residual evals.

    - line_search_shrink (float, (0, 1)):
        Backtracking factor: alpha <- alpha * shrink each failed trial.
        Typical values 0.5 (standard), sometimes 0.8 (gentler) or 0.25 (aggressive).
        If shrink is too close to 1, backtracking may take many reductions.

    - line_search_alpha_min (float, (0, 1]):
        Smallest step length allowed before declaring the Armijo search failed.
        Typical values 1e-6 to 1e-3.
        If too large: you may reject steps that could succeed with more damping.
        If too small: you may waste work taking nearly-zero steps (slow progress).

    - line_search_armijo_c (float, (0, 1)):
        Armijo sufficient-decrease constant in:
          ||r(x - alpha dx)|| <= (1 - c * alpha) ||r(x)||.
        Typical values 1e-4 (common default) to 1e-2.
        Larger c demands more decrease (harder to satisfy), causing more damping.

  Reference:
    Armijo, Larry. 1966. “Minimization of Functions Having Lipschitz Continuous First
    Partial Derivatives.” Pacific Journal of Mathematics 16 (1): 1–3.

  Args:
    get_res_and_tgt_fn: Function that computes the residual and tangents.
    x0: Initial guess, shape (num_dofs,).
    settings: Additional parameters for the linear and nonlinear solvers.
    *params: Extra args forwarded to residual/tangent assembly.
    eps: Small number to avoid division by zero.

  Returns:
    A tuple of the converged (or last) iterate of shape (num_dofs,) and the `NRStats`
      of the solve.
  """
  nl_settings = settings["nonlinear"]
  lin_settings = settings["linear"]

  def safe_norm(r: jnp.ndarray) -> jnp.ndarray:
    n = jnp.linalg.norm(r)
    return jnp.where(jnp.isfinite(n), n, jnp.inf)

  r0, _ = get_res_and_tgt_fn(x0, False, *params)
  r0_norm = jax.lax.stop_gradient(safe_norm(r0))

  state = (x0, r0_norm, jnp.asarray(0, dtype=jnp.int32))

  def cond_fn(state):
    _, r_norm, it = state
    cond_iter = it < nl_settings["max_iter"]
    cond_res = r_norm > _convergence_target(nl_settings, r0_norm)
    return cond_iter & cond_res

  def body_fn(state):
    x, _, it = state

    r, k = get_res_and_tgt_fn(x, True, *params)
    r_norm = jax.lax.stop_gradient(safe_norm(r))

    dx = linear_solve(k, r, params=lin_settings)  # J dx = r

    # Newton predictor alpha0 from half/full trial norms
    x_half = x - 0.5 * dx
    r_half, _ = get_res_and_tgt_fn(x_half, False, *params)
    r_half_norm = jax.lax.stop_gradient(safe_norm(r_half))

    x_full = x - dx
    r_full, _ = get_res_and_tgt_fn(x_full, False, *params)
    r_full_norm = jax.lax.stop_gradient(safe_norm(r_full))

    numer = 3.0 * r_norm + r_full_norm - 4.0 * r_half_norm
    denom = 4.0 * r_norm + 4.0 * r_full_norm - 8.0 * r_half_norm

    alpha0 = jnp.where(jnp.abs(denom) > eps, numer / denom, 1.0)
    alpha0 = jnp.where(jnp.isfinite(alpha0), alpha0, 1.0)
    alpha0 = jnp.clip(alpha0, min=nl_settings["lam_min"], max=1.0)

    # Armijo: ||r(x - a dx)|| <= (1 - c a) ||r(x)||
    def eval_trial(alpha):
      x_trial = x - alpha * dx
      r_trial, _ = get_res_and_tgt_fn(x_trial, False, *params)
      r_trial_norm = jax.lax.stop_gradient(safe_norm(r_trial))
      return x_trial, r_trial_norm

    def armijo_satisfied(alpha, r_trial_norm):
      rhs = (1.0 - nl_settings["line_search_armijo_c"] * alpha) * r_norm
      return jnp.isfinite(r_trial_norm) & (r_trial_norm <= rhs)

    x_trial0, r_trial0_norm = eval_trial(alpha0)
    armijo0 = armijo_satisfied(alpha0, r_trial0_norm)

    # backtrack state:
    bt_state = (
      alpha0,
      x_trial0,
      r_trial0_norm,
      armijo0,
      jnp.asarray(0, dtype=jnp.int32),
      x_trial0,
      r_trial0_norm,
    )

    def backt_cond(st):
      alpha, _, _, armijo, bt_it, *_ = st
      return (
        (~armijo)
        & (bt_it < nl_settings["line_search_max_iter"])
        & (alpha > nl_settings["line_search_alpha_min"])
      )

    def backt_body(st):
      alpha, _, _, _, bt_it, best_x, best_norm = st

      alpha = alpha * nl_settings["line_search_shrink"]
      x_trial, r_trial_norm = eval_trial(alpha)
      armijo = armijo_satisfied(alpha, r_trial_norm)

      improved = r_trial_norm < best_norm
      best_x = jnp.where(improved, x_trial, best_x)
      best_norm = jnp.where(improved, r_trial_norm, best_norm)

      return (alpha, x_trial, r_trial_norm, armijo, bt_it + 1, best_x, best_norm)

    _, x_trial, r_trial_norm, armijo, _, best_x, best_norm = jax.lax.while_loop(
      backt_cond, backt_body, bt_state
    )

    # Accept Armijo if satisfied; else accept best decreasing; else reject.
    best_decreases = jnp.isfinite(best_norm) & (best_norm < r_norm)
    use_best = (~armijo) & best_decreases

    x_new = jnp.where(armijo, x_trial, jnp.where(use_best, best_x, x))
    r_new = jnp.where(armijo, r_trial_norm, jnp.where(use_best, best_norm, r_norm))

    return (x_new, r_new, it + 1)

  x, r_norm, it = jax.lax.while_loop(cond_fn, body_fn, state)
  return x, _nr_stats(nl_settings, it, r_norm, r0_norm)


@_damped_armijo_nr_kernel.defjvp
def solve_jvp_mod(  # noqa: F811
  get_res_and_tgt_fn: Callable,
  settings: Dict,
  primals: tuple[jnp.ndarray, ...],
  tangents: tuple[jnp.ndarray, ...],
) -> tuple[tuple[jnp.ndarray, NRStats], tuple[jnp.ndarray, NRStats]]:
  """JVP rule for the modified Newton-Raphson solver.

  This function implements the JVP rule for the modified Newton-Raphson solver using the
    implicit function theorem. Given the primal inputs (x0, *params) and their tangents,
    we first solve the nonlinear system to obtain the solution x. We then compute the
    product of the derivative of the residual with respect to the parameters times
    the perturbation of the parameters. Finally, we solve the linear system to
    obtain the JVP of the solution with respect to the parameters. The `NRStats` are
    not differentiable and get zero tangents.

  Args:
    get_res_and_tgt_fn: The function to compute the residual and tangents. The function
      takes the same arguments as the primal inputs (x0, *params).
    settings: Additional parameters for the linear and nonlinear solvers.
    primals: A tuple containing the primal inputs (x0, *params).
    tangents: A tuple containing the tangents of the primal inputs (dx0, *dparams).

  Returns: A tuple containing the primal outputs (x, stats) and their tangents
    (dx, zero stats).
  """
  x0, *params = primals
  _, *dparams = tangents

  x_star, stats = _damped_armijo_nr_kernel(get_res_and_tgt_fn, x0, settings, *params)
  _, jacobian = get_res_and_tgt_fn(x_star, True, *params)

  def residual_only(x, *p):
    r, _ = get_res_and_tgt_fn(x, False, *p)
    return r

  _, rhs_vec = jax.jvp(lambda *p: residual_only(x_star, *p), params, dparams)

  dx = linear_solve(jacobian, -rhs_vec, params=settings["linear"])

  dstats = jax.tree.map(jax.custom_derivatives.zero_from_primal, stats)
  return (x_star, stats), (dx, dstats)


def _report_nr_stats(stats: NRStats) -> None:
  """Prints a one-line convergence summary of one or several Newton-Raphson solves.

  Runs on the host via `jax.debug.callback`, so `stats` arrives with concrete numpy
    arrays. Scalar fields describe a single solve. 1D fields (e.g. stacked by
    `jax.lax.scan` over load steps) are summarized over all solves: iteration range,
    final residual and the indices of the solves that did not converge.
    To also write this to a file, replace `print` by a `crops.src.logger` logger here.

  Args:
    stats: `NRStats` with scalar or 1D fields.
  """
  num_iters, converged = np.asarray(stats.num_iters), np.asarray(stats.converged)
  res_norm, init_res_norm = np.asarray(stats.res_norm), np.asarray(stats.init_res_norm)
  rel_res_norm = res_norm / np.maximum(init_res_norm, np.finfo(res_norm.dtype).tiny)

  if num_iters.ndim == 0:
    status = "converged" if converged else "did NOT converge (max_iter reached)"
    print(
      f"NR {status} in {int(num_iters)} iters, |r| = {float(res_norm):.3e}, "
      f"|r|/|r0| = {float(rel_res_norm):.3e}"
    )
    return

  msg = (
    f"NR summary over {num_iters.size} solves: iters min/max/total = "
    f"{num_iters.min()}/{num_iters.max()}/{num_iters.sum()}, "
    f"final |r| = {float(res_norm[-1]):.3e}, |r|/|r0| = {float(rel_res_norm[-1]):.3e}"
  )
  if not converged.all():
    msg += f", unconverged solves: {np.flatnonzero(~converged).tolist()}"
  print(msg)


class NewtonRaphson:
  """Manager class for Nonlinear Solvers.

  Initializes with solver configuration and dispatches solve calls.
  """

  def __init__(
    self,
    solver_type: NonlinearSolvers,
    nonlinear_settings: Dict,
    linear_settings: Dict,
    verbose: bool = True,
  ):
    """Initializes the Newton-Raphson solver manager.

    Attributes:
      solver_type: The type of nonlinear solver to use from `NonlinearSolvers`.
      nonlinear_settings: A dictionary containing settings specific to the nonlinear
      solver, such as maximum iterations and convergence threshold.
      linear_settings: A dictionary containing settings for the linear solver used
      within the nonlinear solver.
      verbose: If `True`, `solve` prints a one-line convergence summary of every solve
        (also from inside `jax.jit` / `jax.lax.scan`). See `report`.
    """
    self.solver_type = solver_type
    self.settings = {"nonlinear": nonlinear_settings, "linear": linear_settings}
    self.verbose = verbose

  def report(self, stats: NRStats) -> None:
    """Prints a convergence summary of `stats` if `self.verbose`, else does nothing.

    Safe to call from traced code. `stats` is either the output of a single `solve`,
      or a stack of them along a leading axis (e.g. the per-step output of a
      `jax.lax.scan` over load steps), in which case one summary line over all solves
      is printed (iteration range, final residual, unconverged solves).

    Args:
      stats: `NRStats` with scalar or 1D fields.
    """
    if self.verbose:
      jax.debug.callback(_report_nr_stats, stats)

  def solve(
    self,
    get_res_and_tgt: Callable[[jnp.ndarray, Any], tuple[jnp.ndarray, Jacobian]],
    x0: jnp.ndarray,
    *params,
  ) -> tuple[jnp.ndarray, NRStats]:
    """Solves the nonlinear system.

    Args:
      get_res_and_tgt: A callable f(x, *params) -> (res, jac).
        Can be manually defined or created via @nonlinear_problem.
      x0: Array of size (n,) of the initial guess.
      *params: Arguments passed to the callable.

    Returns: A tuple of the array of size (n,) of the solution and the `NRStats` of
      the solve (number of iterations, residual norms, convergence flag).
    """

    if self.solver_type == NonlinearSolvers.NEWTON_RAPHSON:
      x, stats = _standard_nr_kernel(get_res_and_tgt, x0, self.settings, *params)

    elif self.solver_type == NonlinearSolvers.MODIFIED_NEWTON_RAPHSON:
      x, stats = _modified_nr_kernel(get_res_and_tgt, x0, self.settings, *params)

    elif self.solver_type == NonlinearSolvers.ARMIJO_BACKTRACKING_NEWTON_RAPHSON:
      x, stats = _damped_armijo_nr_kernel(get_res_and_tgt, x0, self.settings, *params)

    elif self.solver_type == NonlinearSolvers.UNROLLED_NEWTON_RAPHSON:
      x, stats = _unrolled_nr_kernel(get_res_and_tgt, x0, self.settings, *params)

    else:
      raise ValueError(f"Unsupported nonlinear solver type: {self.solver_type}")

    self.report(stats)
    return x, stats


@functools.partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def solve_bisection(residual_fn: Callable, settings: Dict, *params) -> jnp.ndarray:
  """Bisection solver to find the root.

  This function implements the bisection method to find the root of a scalar
    function defined by `residual_fn`. The method iteratively narrows down the interval
    where the root lies until it finds a solution within the specified tolerance.

  Args:
    residual_fn: A callable that takes in a scalar value and additional parameters, and
      returns the residual. The root of this function is what we are trying to find.
    settings: A dictionary containing the settings for the bisection solver, including:
      - "lower": The lower bound of the interval to search for the root.
      - "upper": The upper bound of the interval to search for the root.
      - "tol": The tolerance for convergence. The solver will stop when the width of the
          interval is less than this value.
      - "max_iter": The maximum number of iterations to perform before termination.
    *params: Additional parameters for `residual_fn` when evaluating the residual.

  Returns: The estimated root of the function, a scalar array.
  """
  lower = settings["lower"]
  upper = settings["upper"]
  tol = settings["tol"]
  max_iter = settings["max_iter"]

  def cond_fn(
    state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
  ) -> jnp.ndarray:
    low, up, _, it = state
    return (jnp.abs(up - low) > tol) & (it < max_iter)

  def body_fn(
    state: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
  ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    low, up, f_l, it = state
    mid = 0.5 * (low + up)
    f_mid = residual_fn(mid, *params)

    root_in_upper = jnp.sign(f_mid) == jnp.sign(f_l)

    l_new = jnp.where(root_in_upper, mid, low)
    u_new = jnp.where(root_in_upper, up, mid)
    f_l_new = jnp.where(root_in_upper, f_mid, f_l)

    return (l_new, u_new, f_l_new, it + 1)

  f_lower = residual_fn(lower, *params)
  init_state = (
    jnp.asarray(lower),
    jnp.asarray(upper),
    f_lower,
    jnp.zeros((), dtype=jnp.int32),
  )
  l_f, u_f, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_state)
  return 0.5 * (l_f + u_f)


@solve_bisection.defjvp
def solve_bisection_jvp(
  residual_fn: Callable, settings: Dict, primals, tangents
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Implicit Function Theorem definition for the bisection solver.

  This function implements the JVP rule for the bisection solver using the implicit
    function theorem. Given the primal inputs (the parameters for the residual function)
    and their tangents, we first solve the nonlinear system to obtain the root. We
    then compute the derivative of the residual with respect to the parameters at the
    root, as well as the directional derivative of the residual with respect to the
    parameters. Finally, we apply the implicit function theorem to compute the JVP of
    the root with respect to the parameters.

  Args:
    residual_fn: The function for which we are finding the root. It takes in a scalar
      value and additional parameters, and returns the residual.
    primals: A tuple containing the primal inputs (the parameters for the residual
      function).
    tangents: A tuple containing the tangents of the primal inputs.

  Returns: A tuple containing the root and the JVP of the root with respect to the
    parameters.
  """
  params = primals
  dparams = tangents

  phi_star = solve_bisection(residual_fn, settings, *params)
  df_dphi = jax.grad(residual_fn, argnums=0)(phi_star, *params)
  _, df_dparams_vec = jax.jvp(lambda *p: residual_fn(phi_star, *p), params, dparams)
  dphi = -df_dparams_vec / df_dphi

  return phi_star, dphi
