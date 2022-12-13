from typing import Callable, Tuple, Union

import pandas as pd
import z3

stats = pd.DataFrame(
    [
        {'id': 'A1', 'name': 'total-population', 'count': 7, 'mean': 38.0, 'median': 30.0},
        {'id': 'A2', 'name': 'non-smoker', 'count': 4, 'mean': 33.5, 'median': 30},
        {'id': 'B2', 'name': 'smoker', 'count': 3, 'mean': 44, 'median': 30},
        {'id': 'C2', 'name': 'unemployed', 'count': 4, 'mean': 48.5, 'median': 51},
        {'id': 'D2', 'name': 'employed', 'count': 3, 'mean': 24, 'median': 24},
        {'id': 'A3', 'name': 'single-adults', 'count': None, 'mean': None, 'median': None},
        {'id': 'B3', 'name': 'married-adults', 'count': 5, 'mean': 48, 'median': 36},
        {'id': 'A4', 'name': 'unemployed-non-smoker', 'count': 3, 'mean': 36.67, 'median': 36},
    ]
).set_index('id')

population: range = range(int(stats.filter(items=['A1'], axis=0)['count'][0]))


def _get_stats(id: str, column: str) -> Union[float, int, None]:
    return stats.filter(items=[id], axis=0)[column][0]


def _add_mean_constraint(solver: z3.Solver, *, ages: z3.Array, indices, mean) -> z3.Solver:
    solver.add(z3.Sum([ages[idx] for idx in indices]) / float(len(indices)) == mean)

    return solver


def _add_median_constraint(solver: z3.Solver, *, ages: z3.Array, indices, median) -> z3.Solver:
    med_idx = len(indices) // 2

    if len(indices) % 2 == 0:
        solver.add(ages[indices[med_idx - 1]] + ages[indices[med_idx]] == median * 2)
    else:
        solver.add(z3.Store(ages, indices[med_idx], median) == ages)

    return solver


def reconstruction(
    solver: z3.Solver,
    ages: z3.Array,
    add_min_max_constraint: Callable,
    add_population_mean: Callable,
    add_pairwise_sort_constraint: Callable,
    add_population_median: Callable,
) -> Tuple[z3.Solver, z3.IntVector, z3.IntVector, z3.IntVector]:
    solver = add_min_max_constraint(solver, ages)
    solver = add_pairwise_sort_constraint(solver, ages)
    solver = add_population_median(solver, ages)
    solver = add_population_mean(solver, ages)
    solver, married_indices, _ = _add_marriage_constraints(solver, ages=ages)
    solver, smoker_indices, non_smoker_indices = _add_smoker_constraints(solver, ages=ages)
    solver, employed_indices, __ = _add_employment_constraints(
        solver, ages=ages, non_smoker_indices=non_smoker_indices
    )
    return solver, married_indices, smoker_indices, employed_indices


def _split_pair_of_indices(solver, *, name_pair: Tuple[str, str], first_count: int):
    first_indices = z3.IntVector(name_pair[0], first_count)
    last_indices = z3.IntVector(name_pair[1], len(population) - first_count)

    # indices must between 0 and 7
    solver.add(*[z3.And(idx >= 0, idx < len(population)) for idx in first_indices + last_indices])

    # indices must be distinct
    solver.add(z3.Distinct(*[idx for idx in first_indices + last_indices]))

    # indices must be sorted
    for pair in zip(range(first_count)[:-1], range(first_count)[1:]):
        solver.add(first_indices[pair[0]] < first_indices[pair[1]])

    for pair in zip(range(7 - first_count)[:-1], range(7 - first_count)[1:]):
        solver.add(last_indices[pair[0]] < last_indices[pair[1]])

    return solver, first_indices, last_indices


def _add_marriage_constraints(
    solver: z3.Solver, *, ages: z3.Array
) -> Tuple[z3.Solver, z3.IntVector, z3.IntVector]:
    solver, married_indices, single_indices = _split_pair_of_indices(
        solver, name_pair=('married', 'single'), first_count=int(_get_stats('B3', 'count') or 0)
    )

    # constrain the ages of married people to the legal age
    solver.add(*[ages[idx] >= 18 for idx in married_indices])
    solver.add(*[ages[idx] >= 0 for idx in single_indices])

    # calculate the average for a subset of our database
    solver = _add_mean_constraint(
        solver, ages=ages, indices=married_indices, mean=_get_stats('B3', 'mean')
    )

    # calculate the median for a subset of our database
    solver = _add_median_constraint(
        solver, ages=ages, indices=married_indices, median=_get_stats('B3', 'median')
    )

    # This is the supressed statistic, we know that the count must be 0, 1 or 2
    single_adult_count = [z3.If(ages[idx] >= 18, 1, 0) for idx in single_indices]
    solver.add(z3.Sum(single_adult_count) >= 0)
    solver.add(z3.Sum(single_adult_count) <= 2)

    return solver, married_indices, single_indices


def _add_smoker_constraints(
    solver: z3.Solver, *, ages: z3.Array
) -> Tuple[z3.Solver, z3.IntVector, z3.IntVector]:
    solver, smoker_indices, non_smoker_indices = _split_pair_of_indices(
        solver, name_pair=('smoker', 'non_smoker'), first_count=int(_get_stats('B2', 'count') or 0)
    )

    # add mean constraints
    solver = _add_mean_constraint(
        solver, ages=ages, indices=smoker_indices, mean=_get_stats('B2', 'mean')
    )
    solver = _add_mean_constraint(
        solver, ages=ages, indices=non_smoker_indices, mean=_get_stats('A2', 'mean')
    )

    # add median constraints
    solver = _add_median_constraint(
        solver, ages=ages, indices=smoker_indices, median=_get_stats('B2', 'median')
    )
    solver = _add_median_constraint(
        solver, ages=ages, indices=non_smoker_indices, median=_get_stats('A2', 'median')
    )

    return solver, smoker_indices, non_smoker_indices


def _add_employment_constraints(
    solver: z3.Solver, *, ages: z3.Array, non_smoker_indices
) -> Tuple[z3.Solver, z3.IntVector, z3.IntVector]:
    solver, employed_indices, unemployed_indices = _split_pair_of_indices(
        solver,
        name_pair=('employed', 'unemployed'),
        first_count=int(_get_stats('D2', 'count') or 0),
    )
    solver, unemployed_non_smoker_indices, _ = _split_pair_of_indices(
        solver,
        name_pair=('unemployed_non_smoker', 'rest'),
        first_count=int(_get_stats('A4', 'count') or 0),
    )

    # add mean constraints
    solver = _add_mean_constraint(
        solver, ages=ages, indices=employed_indices, mean=_get_stats('D2', 'mean')
    )
    solver = _add_mean_constraint(
        solver, ages=ages, indices=unemployed_indices, mean=_get_stats('C2', 'mean')
    )

    # add median constraints
    solver = _add_median_constraint(
        solver, ages=ages, indices=employed_indices, median=_get_stats('D2', 'median')
    )
    solver = _add_median_constraint(
        solver, ages=ages, indices=unemployed_indices, median=_get_stats('C2', 'median')
    )

    # intersection of umemployed and non-smoker
    solver.add(
        *[
            z3.And(
                z3.Or(*[i == idx for i in unemployed_indices]),
                z3.Or(*[j == idx for j in non_smoker_indices]),
            )
            for idx in unemployed_non_smoker_indices
        ]
    )

    # add mean constraints
    solver = _add_mean_constraint(
        solver, ages=ages, indices=unemployed_non_smoker_indices, mean=_get_stats('A4', 'mean')
    )

    # add median constraints
    solver = _add_median_constraint(
        solver, ages=ages, indices=unemployed_non_smoker_indices, median=_get_stats('A4', 'median')
    )

    return solver, employed_indices, unemployed_indices
