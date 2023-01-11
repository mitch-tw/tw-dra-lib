from functools import partial
from typing import NewType, Tuple

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


def private_aggregation(
    values: pd.Series,
    budget: float = 5.4,
    epsilons: Tuple[float, ...] = (0.1, 0.1, 0.1),
) -> dict:
    count_epsilon, median_epsilon, mean_epsilon = epsilons
    with dp.BudgetAccountant(epsilon=budget) as accountant:
        return {
            'count': dp.tools.count_nonzero(values, epsilon=count_epsilon, accountant=accountant),
            'median': dp.tools.median(
                values, epsilon=median_epsilon, bounds=(30, 40), accountant=accountant
            ),
            'mean': dp.tools.mean(
                values, epsilon=mean_epsilon, bounds=(30, 40), accountant=accountant
            ),
        }


def global_differential_privacy(df: pd.DataFrame, epsilon: float = 5.4) -> PrivateDataFrame:
    age_bounds = (0, 125)
    clipped = df[df.age.between(*age_bounds)]

    pdf = pd.DataFrame(
        [
            {'name': 'total-population', **private_aggregation(clipped.age)},
            {
                'name': 'non-smoker',
                **private_aggregation(clipped[clipped.smoker == False].age),
            },
            {
                'name': 'smoker',
                **private_aggregation(clipped[clipped.smoker == True].age),
            },
            {
                'name': 'unemployed',
                **private_aggregation(clipped[clipped.employed == False].age),
            },
            {
                'name': 'employed',
                **private_aggregation(clipped[clipped.employed == True].age),
            },
            {
                'name': 'unemployed',
                **private_aggregation(clipped[clipped.employed == False].age),
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
