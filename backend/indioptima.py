from backend.presolver import GenericPresolver
from backend.solver import ADMMSolver
from backend.milp_solver import MILPSolver


def solve(model, backend="cpu", rho=1.0, max_iterations=100):
    if backend != "cpu":
        raise NotImplementedError(
            f"Backend '{backend}' is not currently supported."
        )

    presolver = GenericPresolver()
    presolve_result = presolver.presolve(model)

    model = presolve_result.model

    if model.problem_type == "LP":

        solver = ADMMSolver(
            rho=rho,
            max_iterations=max_iterations
        )

    elif model.problem_type == "MILP":

        solver = MILPSolver(
            rho=rho,
            max_lp_iterations=max_iterations
        )

    else:
        raise NotImplementedError(
            f"Problem type '{model.problem_type}' is not currently supported."
        )

    result = solver.solve(model)

    return result


def main():
    import argparse
    from backend.mps_parser import MPSParser

    parser = argparse.ArgumentParser(
        description="IndiOptima Optimization Solver"
    )

    parser.add_argument(
        "input_file",
        help="Path to an MPS optimization model"
    )

    parser.add_argument(
        "--backend",
        default="cpu",
        choices=["cpu"],
        help="Numerical backend"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Maximum solver iterations"
    )

    args = parser.parse_args()

    model_parser = MPSParser()
    model = model_parser.parse(args.input_file)

    print("IndiOptima")
    print("=" * 40)

    summary = model.summary()

    print(f"Problem type: {summary['problem_type']}")
    print(f"Objective sense: {summary['objective_sense']}")
    print(f"Variables: {summary['variables']}")
    print(f"Constraints: {summary['constraints']}")
    print(f"Nonzeros: {summary['nonzeros']}")
    print(f"Integer variables: {summary['integer_variables']}")
    print(f"Continuous variables: {summary['continuous_variables']}")
    print()

    try:
        result = solve(
            model,
            backend=args.backend,
            max_iterations=args.iterations
        )

        print(f"Status: {result.status}")
        print(f"Objective: {result.objective}")
        print(f"Iterations: {result.iterations}")
        print(f"Constraint violation: {result.constraint_violation}")
        print(f"Bound violation: {result.bound_violation}")
        print(f"Backend: {result.backend}")

    except NotImplementedError as exc:
        print(f"Status: NOT_SUPPORTED")
        print(f"Message: {exc}")


if __name__ == "__main__":
    main()