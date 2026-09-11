import time
import numpy as np

from backend.optimization_model import OptimizationResult


def constraint_violation(Ax, lower, upper):
    lower_violation = np.maximum(lower - Ax, 0)
    upper_violation = np.maximum(Ax - upper, 0)

    return float(
        max(
            np.max(lower_violation),
            np.max(upper_violation)
        )
    )


def bound_violation(x, lower, upper):
    lower_violation = np.maximum(lower - x, 0)
    upper_violation = np.maximum(x - upper, 0)

    return float(
        max(
            np.max(lower_violation),
            np.max(upper_violation)
        )
    )


def integrality_violation(x, integrality):
    integer_mask = integrality != 0

    if not np.any(integer_mask):
        return 0.0

    return float(
        np.max(
            np.abs(x[integer_mask] - np.round(x[integer_mask]))
        )
    )


def verify_solution(
    model,
    solution,
    objective=None,
    feasibility_tolerance=1e-6,
    integrality_tolerance=1e-6
):
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

    feasible = (
        constraint_error <= feasibility_tolerance
        and bound_error <= feasibility_tolerance
        and integer_error <= integrality_tolerance
    )

    return {
        "objective": float(objective),
        "constraint_violation": float(constraint_error),
        "bound_violation": float(bound_error),
        "integrality_violation": float(integer_error),
        "feasible": feasible
    }


class ADMMSolver:

    def __init__(
        self,
        rho=1.0,
        max_iterations=1000,
        tolerance=1e-6,
        backend="cpu"
    ):
        # Keep rho for API compatibility.
        # The new primal-dual method uses adaptive step sizes.
        self.rho = float(rho)
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.backend = backend

        self.objective_history_ = []
        self.constraint_violation_history_ = []
        self.primal_residual_history_ = []
        self.dual_residual_history_ = []

    def _validate_model(self, model):
        if model.problem_type != "LP":
            raise ValueError(
                "ADMMSolver currently supports LP models only. "
                f"Received: {model.problem_type}"
            )

        if model.A.shape[0] != len(model.constraint_lower):
            raise ValueError(
                "Constraint lower bounds do not match A."
            )

        if model.A.shape[0] != len(model.constraint_upper):
            raise ValueError(
                "Constraint upper bounds do not match A."
            )

        if model.A.shape[1] != len(model.objective):
            raise ValueError(
                "Objective length does not match A."
            )

    def _estimate_operator_norm(self, A):
        """
        Estimate ||A||_2 using power iteration.

        This avoids explicitly forming A.T @ A.
        """
        n = A.shape[1]

        rng = np.random.default_rng(42)
        x = rng.standard_normal(n)

        norm_x = np.linalg.norm(x)

        if norm_x == 0:
            return 1.0

        x /= norm_x

        for _ in range(20):
            y = A @ x
            z = A.T @ y

            norm_z = np.linalg.norm(z)

            if norm_z == 0:
                return 1.0

            x = z / norm_z

        Ax = A @ x
        norm_Ax = np.linalg.norm(Ax)

        if norm_Ax == 0:
            return 1.0

        return float(norm_Ax)

    def solve(self, model):

        self._validate_model(model)

        A = model.A
        objective = model.objective
        constraint_lower = model.constraint_lower
        constraint_upper = model.constraint_upper
        variable_lower = model.variable_lower
        variable_upper = model.variable_upper

        n = model.n_variables
        m = model.n_constraints

        self.objective_history_ = []
        self.constraint_violation_history_ = []
        self.primal_residual_history_ = []
        self.dual_residual_history_ = []

        start_time = time.perf_counter()

        # Convert maximization into minimization.
        if model.objective_sense == "min":
            c = objective.copy()
        else:
            c = -objective.copy()

        # ------------------------------------------------------------
        # Initial point
        # ------------------------------------------------------------

        x = np.zeros(n)

        finite_lower = np.isfinite(variable_lower)
        finite_upper = np.isfinite(variable_upper)

        x[finite_lower] = np.maximum(
            x[finite_lower],
            variable_lower[finite_lower]
        )

        x[finite_upper] = np.minimum(
            x[finite_upper],
            variable_upper[finite_upper]
        )

        x_bar = x.copy()

        # Dual variable for Ax belonging to [lower, upper].
        y = np.zeros(m)

        # ------------------------------------------------------------
        # Step sizes
        #
        # For primal-dual hybrid gradient:
        #
        #     tau * sigma * ||A||^2 < 1
        #
        # ------------------------------------------------------------

        operator_norm = self._estimate_operator_norm(A)

        if operator_norm <= 1e-12:
            operator_norm = 1.0

        tau = 0.9 / operator_norm
        sigma = 0.9 / operator_norm

        theta = 1.0

        status = "MAX_ITERATIONS_REACHED"

        final_constraint_violation = np.inf
        final_bound_violation = np.inf

        previous_x = x.copy()
        previous_y = y.copy()

        # ------------------------------------------------------------
        # PDHG iterations
        # ------------------------------------------------------------

        for iteration in range(1, self.max_iterations + 1):

            # --------------------------------------------------------
            # Dual update
            #
            # We need:
            #
            #     y = prox_{sigma f*}(y + sigma A x_bar)
            #
            # where f is the indicator function of:
            #
            #     lower <= z <= upper
            #
            # The conjugate proximal operator can be written through
            # Moreau's identity:
            #
            #     prox_{sigma f*}(v)
            #       = v - sigma * projection(v/sigma)
            #
            # --------------------------------------------------------

            y_previous = y.copy()

            v = y + sigma * (A @ x_bar)

            z = np.minimum(
                np.maximum(v / sigma, constraint_lower),
                constraint_upper
            )

            # Handle infinite bounds correctly.
            finite_lower_constraint = np.isfinite(constraint_lower)
            finite_upper_constraint = np.isfinite(constraint_upper)

            z = v / sigma

            z[finite_lower_constraint] = np.maximum(
                z[finite_lower_constraint],
                constraint_lower[finite_lower_constraint]
            )

            z[finite_upper_constraint] = np.minimum(
                z[finite_upper_constraint],
                constraint_upper[finite_upper_constraint]
            )

            y = v - sigma * z

            # --------------------------------------------------------
            # Primal update
            #
            # x = projection_bounds(
            #       x - tau * (c + A.T y)
            #     )
            # --------------------------------------------------------

            x_previous = x.copy()

            x = x - tau * (c + A.T @ y)

            # Project onto variable bounds.
            x = np.maximum(x, variable_lower)
            x = np.minimum(x, variable_upper)

            # --------------------------------------------------------
            # Extrapolation
            # --------------------------------------------------------

            x_bar = x + theta * (x - x_previous)

            # --------------------------------------------------------
            # Diagnostics
            # --------------------------------------------------------

            Ax = A @ x

            current_constraint_violation = constraint_violation(
                Ax,
                constraint_lower,
                constraint_upper
            )

            current_bound_violation = bound_violation(
                x,
                variable_lower,
                variable_upper
            )

            primal_residual = np.linalg.norm(
                x - x_previous
            )

            dual_residual = np.linalg.norm(
                y - y_previous
            )

            original_objective = objective @ x

            self.objective_history_.append(
                float(original_objective)
            )

            self.constraint_violation_history_.append(
                float(current_constraint_violation)
            )

            self.primal_residual_history_.append(
                float(primal_residual)
            )

            self.dual_residual_history_.append(
                float(dual_residual)
            )

            # --------------------------------------------------------
            # Convergence
            # --------------------------------------------------------

            if (
                current_constraint_violation <= self.tolerance
                and current_bound_violation <= self.tolerance
                and primal_residual <= self.tolerance
                and dual_residual <= self.tolerance
            ):
                status = "CONVERGED"

                final_constraint_violation = (
                    current_constraint_violation
                )

                final_bound_violation = (
                    current_bound_violation
                )

                break

        else:
            final_constraint_violation = (
                self.constraint_violation_history_[-1]
            )

            final_bound_violation = bound_violation(
                x,
                variable_lower,
                variable_upper
            )

        # ------------------------------------------------------------
        # Final verification
        # ------------------------------------------------------------

        final_objective = objective @ x

        verification = verify_solution(
            model,
            x,
            final_objective
        )

        solve_time = time.perf_counter() - start_time

        if status == "CONVERGED":

            if verification["feasible"]:
                status = "FEASIBLE"
            else:
                status = "MAX_ITERATIONS_REACHED"

        return OptimizationResult(
            status=status,
            objective=float(final_objective),
            solution=x,
            solve_time=float(solve_time),
            iterations=iteration,
            constraint_violation=float(
                verification["constraint_violation"]
            ),
            bound_violation=float(
                verification["bound_violation"]
            ),
            backend=self.backend,
            problem_type=model.problem_type
        )