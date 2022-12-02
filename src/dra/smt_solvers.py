import numpy as np
import z3
from pipe import map


def _check_length(model: z3.ModelRef) -> z3.ModelRef:
    model_vars = model.decls()
    n_vars = len(model_vars)

    if n_vars != 1:
        raise ValueError(f'Model should only contain one variable, found {n_vars}')
    return model


def _check_scores_match_expected(model: z3.ModelRef) -> z3.ModelRef:
    model_vars = model.decls()
    results = [model.evaluate(z3.Select(model[model_vars[0]], i)) for i in range(10)]

    # converting from z3 object to int has to be done via string conversion first
    vals = [int(str(v)) for v in results]
    expected_mean = 76
    calculated_mean = np.mean(vals)

    if calculated_mean != expected_mean:
        raise ValueError(
            f'expected mean is {expected_mean}. Your model returns a mean of {calculated_mean}'
        )

    if min(vals) < 0:
        raise ValueError('all values must be above or equal to 0')

    if min(vals) > 100:
        raise ValueError('all values must be below or equal to 100')
    return model


def check_results(model: z3.ModelRef) -> bool:
    list((model,) | map(_check_length) | map(_check_scores_match_expected))
    print("Nice! You've generated the right model for this module ✅🎉😁")
    return True
