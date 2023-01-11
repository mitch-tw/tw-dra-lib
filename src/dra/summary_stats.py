import pandas as pd


def generate_block_stats(*dfs: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    for df in dfs:
        block_stats = (
            df.reset_index()
            .set_index(['id', 'name', 'index'])
            .unstack()
            .reset_index()
            .droplevel(0, axis=1)
        )
        block_stats.columns = ['id', 'name', 'count', 'mean', 'median']
        out = pd.concat([out, block_stats])
    out['mean'] = out['mean'].round(2)
    return out.reset_index(drop=True)


expected_results = pd.DataFrame(
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
)

database = pd.DataFrame(
    [
        ('Sara Gray', 8, False, False, False),
        ('Joseph Collins', 18, False, True, True),
        ('Vincent Porter', 24, True, False, True),
        ('Tiffany Brown', 30, True, True, True),
        ('Brenda Small', 36, True, False, False),
        ('Dr. Tina Ayala', 66, True, False, False),
        ('Rodney Gonzalez', 84, True, True, False),
    ],
    columns=['name', 'age', 'married', 'smoker', 'employed'],
)


def check_results(generated: pd.DataFrame) -> bool:
    results = generated.compare(expected_results)
    if len(results) > 0:
        results['stat_name'] = [expected_results.iloc[i]['name'] for i in results.index]
        print(results)
        raise ValueError("Oops! Looks like you've generated the wrong results")
    print("Nice! You've generated the right summary statistics for this module ✅🎉😁")
    return True
