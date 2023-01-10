import pytest
import z3

from dra.reconstruction_solver import (
    add_min_max_constraint,
    add_pairwise_sort_constraint,
    add_population_mean,
    add_population_median,
    model_as_dataframe,
    reconstruction,
)


@pytest.fixture
def reconstructed():
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
    output.age = output.age.astype(str).astype(int)
    return output


@pytest.fixture
def employed(reconstructed):
    return reconstructed[reconstructed.employed == True]


@pytest.fixture
def unemployed(reconstructed):
    return reconstructed[reconstructed.employed == False]


@pytest.fixture
def married(reconstructed):
    return reconstructed[reconstructed.married == True]


@pytest.fixture
def single(reconstructed):
    return reconstructed[reconstructed.married == False]


@pytest.fixture
def smokers(reconstructed):
    return reconstructed[reconstructed.smoker == True]


@pytest.fixture
def non_smokers(reconstructed):
    return reconstructed[reconstructed.smoker == False]


@pytest.fixture
def unemployed_non_smokers(reconstructed):
    return reconstructed[(reconstructed.smoker == False) & (reconstructed.employed == False)]
