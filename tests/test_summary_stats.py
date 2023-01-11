import pandas as pd
import pytest

from dra.summary_stats import check_results, database, expected_results, generate_block_stats


def test_database():
    assert isinstance(database, pd.DataFrame)
    assert len(database) == 7
    assert database.columns.tolist() == ['name', 'age', 'married', 'smoker', 'employed']


def test_expected_results():
    assert isinstance(expected_results, pd.DataFrame)
    assert len(expected_results) == 8
    assert list(expected_results.id.unique()) == ['A1', 'A2', 'B2', 'C2', 'D2', 'A3', 'B3', 'A4']


def test_check_results_errors():
    wrong_output = pd.DataFrame(
        [
            {'id': 'A1', 'name': 'total-population', 'count': 6, 'mean': 38.0, 'median': 30.0},
            {'id': 'A2', 'name': 'non-smoker', 'count': 4, 'mean': 33.5, 'median': 30},
            {'id': 'B2', 'name': 'smoker', 'count': 3, 'mean': 55, 'median': 30},
            {'id': 'C2', 'name': 'unemployed', 'count': 4, 'mean': 48.5, 'median': 51},
            {'id': 'D2', 'name': 'employed', 'count': 3, 'mean': 24, 'median': 24},
            {'id': 'A3', 'name': 'single-adults', 'count': None, 'mean': None, 'median': None},
            {'id': 'B3', 'name': 'married-adults', 'count': 5, 'mean': 48, 'median': 36},
            {'id': 'A4', 'name': 'unemployed-non-smoker', 'count': 3, 'mean': 36.67, 'median': 36},
        ]
    )
    with pytest.raises(ValueError):
        check_results(wrong_output)


def test_check_results():
    passed = check_results(expected_results)
    assert passed is True


def test_generate_block_stats():
    total_population = database.agg({'age': ['count', 'mean', 'median']}).assign(
        name='total-population', id='A1'
    )
    non_smoker = (
        database[database.smoker == False]  # noqa: E712
        .agg({'age': ['count', 'mean', 'median']})
        .assign(name='total-population', id='A2')
    )
    out = generate_block_stats(total_population, non_smoker)
    assert out.columns.tolist() == ['id', 'name', 'count', 'mean', 'median']
    assert len(out) == 2
