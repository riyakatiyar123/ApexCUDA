import time
import numpy as np
import scipy.sparse as sp

from scipy.sparse.linalg import factorized

from backend.optimization_model import OptimizationResult

def constraint_violation(Ax, lower, upper):
    lower_violation = np.maximum(lower - Ax, 0)
    upper_violation = np.maximum(Ax - upper, 0)

    max_lower = np.max(lower_violation)
    max_upper = np.max(upper_violation)

    return max(max_lower, max_upper)


def bound_violation(x, lower, upper):
    lower_violation = np.maximum(lower - x, 0)
    upper_violation = np.maximum(x - upper, 0)

    return max(
        np.max(lower_violation),
        np.max(upper_violation)
    )


def integrality_violation(x, integrality):
    integer_mask = integrality != 0

    if not np.any(integer_mask):
        return 0.0

    return np.max(
        np.abs(x[integer_mask] - np.round(x[integer_mask]))
    )


def verify_solution(model, solution, objective=None):
    """
    Verify a solution against a generic OptimizationModel.
    """

    Ax = model.A @ solution

    constraint_error = constraint_violation(
        Ax,
        model.constraint_lower,
        model.constraint_upper
    )

    bound_error = bound_violation(
        solution,
        model.variable_lower,
        model.variable_upper
    )

    integer_error = integrality_violation(
        solution,
        model.variable_integrality
    )

    if objective is None:
        objective = model.objective @ solution

    return {
        "objective": float(objective),
        "constraint_violation": float(constraint_error),
        "bound_violation": float(bound_error),
        "integrality_violation": float(integer_error),
        "feasible": (
            constraint_error == 0.0
            and bound_error == 0.0
            and integer_error == 0.0
        )
    }

def admm_lp_test(
    A,
    c,
    lower,
    upper,
    var_lower,
    var_upper,
    rho=1.0,
    iterations=100
):
    n = len(c)

    x = np.zeros(n)
    z = A @ x
    u = np.zeros(len(z))

    objectives = []
    violations = []

    ATA = A.T @ A
    M = sp.eye(n) + rho * ATA

    solve_M = factorized(M.tocsc())

    for k in range(iterations):

        rhs = -c + rho * A.T @ (z - u)

        x = solve_M(rhs)

        x = np.clip(
            x,
            var_lower,
            var_upper
        )

        Ax = A @ x

        z = np.minimum(
            np.maximum(Ax + u, lower),
            upper
        )

        u = u + Ax - z

        objective = c @ x

        violation = constraint_violation(
            Ax,
            lower,
            upper
        )

        objectives.append(objective)
        violations.append(violation)

    return x, objectives, violations

class ADMMSolver:

    def __init__(self, rho=1.0, max_iterations=100):
        self.rho = rho
        self.max_iterations = max_iterations

    def solve(self, model):
        if model.problem_type != "LP":
            raise NotImplementedError(
                f"ADMMSolver currently supports LP only. Received: {model.problem_type}"
            )

        start_time = time.time()

        # ADMM currently minimizes the objective.
        # For a maximization problem, minimize the negative objective.
        solver_objective = model.objective.copy()

        if model.objective_sense == "max":
            solver_objective = -solver_objective

        x, objectives, violations = admm_lp_test(
            model.A,
            solver_objective,
            model.constraint_lower,
            model.constraint_upper,
            model.variable_lower,
            model.variable_upper,
            rho=self.rho,
            iterations=self.max_iterations
        )

        solve_time = time.time() - start_time

        # Convert the internal objective back to the model's objective.
        actual_objective = model.objective @ x

        verification = verify_solution(
            model,
            x,
            actual_objective
        )

        status = "FEASIBLE" if verification["feasible"] else "ERROR"

        return OptimizationResult(
            status=status,
            objective=verification["objective"],
            solution=x,
            solve_time=solve_time,
            iterations=self.max_iterations,
            constraint_violation=verification["constraint_violation"],
            bound_violation=verification["bound_violation"],
            backend="cpu",
            problem_type=model.problem_type
        )