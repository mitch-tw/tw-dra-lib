from functools import partial
from typing import NewType, Tuple

import altair as alt
import diffprivlib as dp
import numpy as np
import pandas as pd
from faker import Faker

PrivateDataFrame = NewType('PrivateDataFrame', pd.DataFrame)  # type: ignore

faker = Faker()
rand_bool = partial(np.random.choice, (True, False))


class Colours:
    red = '#f2617a'
    orange = '#cc850a'
    green = '#6b9e78'
    teal = '#47a1ad'
    purple = '#634f7d'
    dark_blue = '#003d4f'
    light_gray = '#edf1f3'


def epsilon_ranges(
    values: list,
    start: float = 0.01,
    stop: float = 1.0,
    step: float = 0.01,
    bounds: Tuple[int, int] = (1, 125),
):
    return pd.DataFrame(
        [
            {
                'i': i,
                'noisy_mean': dp.tools.mean(values, epsilon=i, bounds=bounds),
                'actual_mean': np.mean(values),
                'noisy_median': dp.tools.median(values, epsilon=i, bounds=bounds),
                'actual_median': np.median(values),
                'noisy_count': dp.tools.count_nonzero(values, epsilon=i),
                'actual_count': len(values),
            }
            for i in np.arange(start, stop, step)
        ]
    )


def line_chart(
    df: pd.DataFrame,
    title: str,
    subtitle: str,
    x: str | None = None,
    y: str | None = None,
    y2: str | None = None,
    color: str = Colours.red,
) -> alt.Chart:
    dp_chart = (
        alt.Chart(df)
        .mark_line(size=2, point=True)
        .encode(
            x=x,
            y=alt.Y(y, scale=alt.Scale(domain=(df[y].min(), df[y].max()))),
            color=alt.value(color),
            tooltip=list(df.columns),
        )
        .properties(
            width=800,
            title=alt.TitleParams(
                text=title,
                subtitle=subtitle,
                anchor='start',
                dy=-10,
                fontSize=18,
                subtitleFontSize=15,
            ),
        )
    )
    base_chart = (
        alt.Chart(df)
        .mark_rule()
        .encode(y=alt.Y(y2, scale=alt.Scale(domain=(df[y2].min(), df[y2].max()))))
    )
    return base_chart + dp_chart


def make_database(i: int = 100):
    database = pd.DataFrame(
        [
            {
                'age': np.random.randint(0, 125),
                'name': faker.name(),
                'married': rand_bool(),
                'smoker': rand_bool(),
                'employed': rand_bool(),
            }
            for _ in range(i)
        ],
        columns=['name', 'age', 'married', 'smoker', 'employed'],
    )
    return database


def randomised_response(truth: bool) -> bool:
    if np.random.choice((True, False)):
        return truth
    else:
        return np.random.choice((True, False))


def private_aggregation(
    values: pd.Series,
    budget: float = 5.4,
    epsilons: Tuple[float, ...] = (0.1, 0.1, 0.1),
    bounds: Tuple[int, int] = (0, 250),
) -> dict:
    count_epsilon, median_epsilon, mean_epsilon = epsilons
    with dp.BudgetAccountant(epsilon=budget) as accountant:
        return {
            'count': dp.tools.count_nonzero(values, epsilon=count_epsilon, accountant=accountant),
            'median': dp.tools.median(
                values, epsilon=median_epsilon, bounds=bounds, accountant=accountant
            ),
            'mean': dp.tools.mean(
                values, epsilon=mean_epsilon, bounds=bounds, accountant=accountant
            ),
        }


def global_differential_privacy(
    df: pd.DataFrame,
    budget: float = 5.4,
    epsilons: Tuple[float, ...] = (0.1, 0.1, 0.1),
    age_bounds: Tuple[int, int] = (0, 125),
) -> PrivateDataFrame:
    clipped = df[df.age.between(*age_bounds)]

    pdf = pd.DataFrame(
        [
            {'name': 'total-population', **private_aggregation(clipped.age, budget, epsilons)},
            {
                'name': 'non-smoker',
                **private_aggregation(clipped[clipped.smoker == False].age, budget, epsilons),
            },
            {
                'name': 'smoker',
                **private_aggregation(clipped[clipped.smoker == True].age, budget, epsilons),
            },
            {
                'name': 'unemployed',
                **private_aggregation(clipped[clipped.employed == False].age, budget, epsilons),
            },
            {
                'name': 'employed',
                **private_aggregation(clipped[clipped.employed == True].age, budget, epsilons),
            },
            {
                'name': 'unemployed',
                **private_aggregation(clipped[clipped.employed == False].age, budget, epsilons),
            },
        ]
    )
    return PrivateDataFrame(pdf)


def local_differential_privacy(df: pd.DataFrame, epsilon: float = 0.33) -> PrivateDataFrame:
    laplace = dp.mechanisms.Laplace(epsilon=epsilon, sensitivity=1)
    private_database = df.copy()
    private_database = private_database.drop(columns=['name'])
    private_database['age'] = df.age.apply(lambda val: int(laplace.randomise(val)))
    private_database['married'] = df.married.apply(randomised_response)
    private_database['smoker'] = df.smoker.apply(randomised_response)
    private_database['employed'] = df.employed.apply(randomised_response)
    return PrivateDataFrame(private_database)


def no_differential_privacy(database: pd.DataFrame) -> pd.DataFrame:
    def agg(ages: pd.Series) -> dict:
        return {'count': len(ages), 'median': np.median(ages), 'mean': np.mean(ages)}

    return pd.DataFrame(
        [
            {'name': 'total-population', **agg(database.age)},
            {
                'name': 'non-smoker',
                **agg(database[database.smoker == False].age),
            },
            {
                'name': 'smoker',
                **agg(database[database.smoker == True].age),
            },
            {
                'name': 'unemployed',
                **agg(database[database.employed == False].age),
            },
            {
                'name': 'employed',
                **agg(database[database.employed == True].age),
            },
            {
                'name': 'unemployed',
                **agg(database[database.employed == False].age),
            },
        ]
    )
