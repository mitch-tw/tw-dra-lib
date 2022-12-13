from functools import partial
from typing import NewType

import diffprivlib as dp
import numpy as np
import pandas as pd
from faker import Faker

PrivateDataFrame = NewType('PrivateDataFrame', pd.DataFrame)  # type: ignore

faker = Faker()
rand_bool = partial(np.random.choice, (True, False))


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


def private_aggregation(values: pd.Series) -> dict:
    with dp.BudgetAccountant(epsilon=1.1) as budget:
        return {
            'count': dp.tools.count_nonzero(values, epsilon=0.1, accountant=budget),
            'median': dp.tools.median(values, epsilon=0.5, bounds=(30, 40), accountant=budget),
            'mean': dp.tools.mean(values, epsilon=0.5, bounds=(30, 40), accountant=budget),
        }


def global_differential_privacy(df: pd.DataFrame) -> PrivateDataFrame:
    age_bounds = (0, 125)
    clipped = df[df.age.between(*age_bounds)]

    pdf = pd.DataFrame(
        [
            {'name': 'total-population', **private_aggregation(clipped.age)},
            {
                'name': 'non-smoker',
                **private_aggregation(clipped[clipped.smoker == False].age),  # noqa: E712
            },
            {
                'name': 'smoker',
                **private_aggregation(clipped[clipped.smoker == True].age),  # noqa: E712
            },
            {
                'name': 'unemployed',
                **private_aggregation(clipped[clipped.employed == False].age),  # noqa: E712
            },
            {
                'name': 'employed',
                **private_aggregation(clipped[clipped.employed == True].age),  # noqa: E712
            },
            {
                'name': 'unemployed',
                **private_aggregation(clipped[clipped.employed == False].age),  # noqa: E712
            },
        ]
    )
    return PrivateDataFrame(pdf)


def local_differential_privacy(df: pd.DataFrame) -> PrivateDataFrame:
    private_database = df.copy()
    private_database = private_database.drop(columns=['name'])
    private_database['age'] = df.age.apply(
        lambda val: int(dp.mechanisms.Laplace(epsilon=0.33, sensitivity=1).randomise(val))
    )
    private_database['married'] = df.married.apply(randomised_response)
    private_database['smoker'] = df.smoker.apply(randomised_response)
    private_database['employed'] = df.employed.apply(randomised_response)
    return PrivateDataFrame(private_database)


def main():
    database = make_database(10)
    global_pdf: PrivateDataFrame = global_differential_privacy(database)
    local_pdf: PrivateDataFrame = local_differential_privacy(database)
    print(global_pdf)
    print('\n')
    print(local_pdf)
