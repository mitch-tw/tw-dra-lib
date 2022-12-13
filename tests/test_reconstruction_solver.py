from itertools import tee

import z3

from dra.reconstruction_solver import check_accuracy, database, model_as_dataframe, reconstruction


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def add_min_max_constraint(solver: z3.Solver, ages: z3.Array) -> z3.Solver:
    for i in range(7):
        solver.add(ages[i] >= 0)
        solver.add(ages[i] <= 125)

    return solver


def add_population_mean(solver: z3.Solver, ages: z3.Array) -> z3.Solver:
    solver.add(z3.Sum([ages[i] for i in range(7)]) / 7 == 38)
    return solver


def add_pairwise_sort_constraint(solver: z3.Solver, ages: z3.Array) -> z3.Solver:
    population = range(7)
    for a, b in pairwise([ages[i] for i in population]):
        solver.add(a <= b)
    return solver


def add_population_median(solver: z3.Solver, ages: z3.Array) -> z3.Solver:
    solver.add(z3.Select(ages, 3) == 30)
    return solver


def test_reconstruction():
    solver = z3.Solver()
    ages = z3.Array('ages', z3.IntSort(), z3.IntSort())
    solver, married_indices, smoker_indices, employed_indices = reconstruction(
        solver,
        ages=ages,
        add_min_max_constraint=add_min_max_constraint,
        add_population_median=add_population_median,
        add_population_mean=add_population_mean,
        add_pairwise_sort_constraint=add_pairwise_sort_constraint,
    )

    assert z3.sat == solver.check()

    model = solver.model()
    output = model_as_dataframe(
        model,
        ages=ages,
        married_indices=married_indices,
        smoker_indices=smoker_indices,
        employed_indices=employed_indices,
    )
    assert float(check_accuracy(output, database).replace('%', '')) >= 92
