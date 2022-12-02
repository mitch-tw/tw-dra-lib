import pytest
import z3

from dra.smt_solvers import check_results


def _create_model_solution(
    average_score: int, n_responses: int, min_: int = 0, max_: int = 100
) -> z3.ModelRef:
    solver = z3.Solver()

    scores = z3.Array('scores', z3.IntSort(), z3.IntSort())

    solver.add(
        z3.Sum([z3.Select(scores, i) for i in range(n_responses)]) / n_responses == average_score
    )

    solver.add(*[z3.Select(scores, i) <= max_ for i in range(n_responses)])
    solver.add(*[z3.Select(scores, i) >= min_ for i in range(n_responses)])
    solver.check()
    model = solver.model()
    return model


def _create_model_with_too_many_variables() -> z3.ModelRef:
    solver = z3.Solver()

    scores = z3.Array('scores', z3.IntSort(), z3.IntSort())
    extra = z3.Int('extra')

    solver.add(z3.Sum([z3.Select(scores, i) for i in range(3)]) == 5)

    solver.add(extra == 1)
    solver.check()
    model = solver.model()
    return model


def _create_model_without_bounds(average_score: int, n_responses: int) -> z3.ModelRef:
    solver = z3.Solver()

    scores = z3.Array('scores', z3.IntSort(), z3.IntSort())

    solver.add(
        z3.Sum([z3.Select(scores, i) for i in range(n_responses)]) / n_responses == average_score
    )
    solver.check()
    model = solver.model()
    return model


def _create_model_with_no_vars() -> z3.ModelRef:
    solver = z3.Solver()
    solver.check()
    return solver.model()


def test_statistical_solver_checker():
    model = _create_model_solution(76, 10)
    passed = check_results(model)
    assert passed is True


def test_wrong_mean():
    model = _create_model_solution(6, 10)
    with pytest.raises(ValueError):
        check_results(model)


def test_wrong_bounds():
    model = _create_model_without_bounds(76, 10)
    with pytest.raises(ValueError):
        check_results(model)


def test_too_many_vars():
    model = _create_model_with_too_many_variables()
    with pytest.raises(ValueError):
        check_results(model)


def test_no_vars():
    model = _create_model_with_no_vars()
    with pytest.raises(ValueError):
        check_results(model)
