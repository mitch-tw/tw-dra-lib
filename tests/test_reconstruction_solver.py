from itertools import tee

import pandas as pd
import z3

from dra.reconstruction_solver import reconstruction


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


database = pd.DataFrame(
    [
        (8, False, False, False),
        (18, False, True, True),
        (24, True, False, True),
        (30, True, True, True),
        (36, True, False, False),
        (66, True, False, False),
        (84, True, True, False),
    ],
    columns=['age', 'married', 'smoker', 'employed'],
)


def add_min_max_constraint(solver: z3.Solver, ages: z3.Array) -> z3.Solver:
    for i in range(7):
        solver.add(ages[i] > 0)
        solver.add(ages[i] < 100)

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


def check_accuracy(output, database) -> float:
    match, non_match = 0, 0
    computed = [tuple(v.values()) for v in output.to_dict(orient='records')]  # type: ignore
    original = [tuple(v.values()) for v in database.to_dict(orient='records')]  # type: ignore

    to_check = [list(zip(computed[i], original[i])) for i in range(7)]
    for items in to_check:
        for pair in items:
            if pair[0] == pair[1]:
                match += 1
            else:
                non_match += 1
    return (match / (match + non_match)) * 100


def model_as_dataframe(
    model, *, ages, married_indices, smoker_indices, employed_indices
) -> pd.DataFrame:
    if isinstance(model, z3.ModelRef):
        df = pd.DataFrame(
            {
                'age': [model.evaluate(z3.Select(ages, i)) for i in range(7)],
                'married': [i in [model[idx] for idx in married_indices] for i in range(7)],
                'smoker': [i in [model[idx] for idx in smoker_indices] for i in range(7)],
                'employed': [i in [model[idx] for idx in employed_indices] for i in range(7)],
            }
        )

        return df
    raise TypeError(
        'model is not initialised. Either there is no valid model, '
        'or you forgot to run DatabaseReconstructionAttack.run?'
    )


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
    assert check_accuracy(output, database) >= 92
