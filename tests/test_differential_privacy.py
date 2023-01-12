import pandas as pd

from dra.differential_privacy import epsilon_ranges


def test_epsilon_ranges_dataframe_result():
    result = epsilon_ranges([i for i in range(100)])
    assert isinstance(result, pd.DataFrame)


def test_epsilon_ranges_columns():
    result = epsilon_ranges([i for i in range(100)])
    columns = list(result.columns)
    assert columns == [
        'epsilon',
        'noisy_mean',
        'actual_mean',
        'noisy_median',
        'actual_median',
        'noisy_count',
        'actual_count',
    ]


def test_epsilon_ranges_mean_changes():
    values = [i for i in range(100)]
    result = epsilon_ranges(values)

    assert all(result.apply(lambda row: row.actual_mean != row.noisy_mean, axis=1).values)


def test_epsilon_ranges_median_changes():
    values = [i for i in range(100)]
    result = epsilon_ranges(values)

    assert all(result.apply(lambda row: row.actual_median != row.noisy_median, axis=1).values)
