from dra.summary_stats import database, expected_results, check_results, generate_block_stats
import pandas as pd
import pytest


def test_database():
    assert isinstance(database, pd.DataFrame)
    assert len(database) == 7
    assert database.columns.tolist() == ["age", "married", "smoker", "employed"]


def test_expected_results():
    assert isinstance(expected_results, pd.DataFrame)
    assert len(expected_results) == 8
    assert list(expected_results.id.unique()) == ['A1', 'A2', 'B2', 'C2', 'D2', 'A3', 'B3', 'A4']


def test_check_results_errors():
    with pytest.raises(ValueError):
        check_results(pd.DataFrame())


def test_check_results():
    passed = check_results(expected_results)
    assert passed is True


def test_generate_block_stats():
    total_population = (
        database
        .agg({"age": ["count", "mean", "median"]})
        .assign(name="total-population", id="A1")
    )
    non_smoker = (
        database
        [database.smoker == False]
        .agg({"age": ["count", "mean", "median"]})
        .assign(name="total-population", id="A2")
    )
    out = generate_block_stats(total_population, non_smoker)
    assert out.columns.tolist() == ['id', 'name', 'count', 'mean', 'median']
    assert len(out) == 2
