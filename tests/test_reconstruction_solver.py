from math import floor

import pandas as pd
import z3

from dra.reconstruction_solver import (
    add_min_max_constraint,
    add_pairwise_sort_constraint,
    add_population_mean,
    add_population_median,
    check_accuracy,
    database,
    model_as_dataframe,
    reconstruction,
    stats,
)


def _get_stat(name: str, column: str):
    return stats.filter(items=[name], axis=0)[column][0]


def test_reconstruction():
    solver = z3.Solver()
    ages = z3.Array('ages', z3.IntSort(), z3.IntSort())
    solver, married_indices, smoker_indices, employed_indices = reconstruction(
        solver,
        ages=ages,
        min_max_constraint=add_min_max_constraint,
        population_median=add_population_median,
        population_mean=add_population_mean,
        pairwise_sort_constraint=add_pairwise_sort_constraint,
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


def test_population_count(reconstructed):
    assert len(reconstructed.age) == _get_stat('A1', 'count')


def test_population_mean(reconstructed):
    assert floor(reconstructed.age.mean()) == _get_stat('A1', 'mean')


def test_population_median(reconstructed: pd.DataFrame):
    assert floor(reconstructed.age.median()) == _get_stat('A1', 'median')


def test_employment_count(employed, unemployed):
    assert len(employed) == _get_stat('D2', 'count')
    assert len(unemployed) == _get_stat('C2', 'count')


def test_employment_mean(employed, unemployed):
    assert floor(employed.age.mean()) == _get_stat('D2', 'mean')
    assert unemployed.age.mean() == _get_stat('C2', 'mean')


def test_employment_median(employed, unemployed):
    assert floor(employed.age.median()) == _get_stat('D2', 'median')
    assert floor(unemployed.age.median()) == _get_stat('C2', 'median')


def test_marriages_count(married, single):
    assert len(married) == _get_stat('B3', 'count')
    assert len(single) > 0
    assert len(single) <= 3


def test_marriages_mean(married, single):
    assert floor(married.age.mean()) == _get_stat('B3', 'mean')


def test_marriages_median(married, single):
    assert floor(married.age.median()) == _get_stat('B3', 'median')


def test_smokers_count(smokers, non_smokers):
    assert len(smokers) == _get_stat('B2', 'count')
    assert len(non_smokers) == _get_stat('A2', 'count')


def test_smokers_mean(smokers, non_smokers):
    assert floor(smokers.age.mean()) == _get_stat('B2', 'mean')
    assert non_smokers.age.mean() == _get_stat('A2', 'mean')


def test_smokers_median(smokers, non_smokers):
    assert floor(smokers.age.median()) == _get_stat('B2', 'median')
    assert floor(non_smokers.age.median()) == _get_stat('A2', 'median')


def test_unemployed_non_smoker_count(unemployed_non_smokers):
    assert len(unemployed_non_smokers) == _get_stat('A4', 'count')


def test_unemployed_non_smoker_mean(unemployed_non_smokers):
    assert floor(unemployed_non_smokers.age.mean()) >= _get_stat('A4', 'mean') - 1
    assert floor(unemployed_non_smokers.age.mean()) <= _get_stat('A4', 'mean')


def test_unemployed_non_smoker_median(unemployed_non_smokers):
    assert floor(unemployed_non_smokers.age.median()) == _get_stat('A4', 'median')
