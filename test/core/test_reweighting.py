import os

import numpy as np
import pandas as pd
import pytest
from bilby.core.likelihood import GaussianLikelihood
from bilby.core.prior import PriorDict, Uniform
from bilby.core.result import Result, get_weights_for_reweighting


def _dummy_func(x):
    return x


def setup_function():
    pass


def teardown_function():
    if os.path.exists("resume_file.pkl"):
        os.remove("resume_file.pkl")


@pytest.mark.parametrize("npool", [1, 2])
def test_reweighting(npool):
    import multiprocessing

    try:
        multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass
    old_likelihood = GaussianLikelihood(
        x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 10), func=_dummy_func, sigma=1
    )
    new_likelihood = GaussianLikelihood(
        x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 10), func=_dummy_func, sigma=2
    )
    old_priors = PriorDict({"a": Uniform(0, 1)})
    new_priors = PriorDict({"a": Uniform(0, 2)})
    result = Result(posterior=pd.DataFrame(old_priors.sample(10)))
    resume_file = "resume_file.pkl"
    (
        ln_weights,
        new_log_likelihood_array,
        new_log_prior_array,
        old_log_likelihood_array,
        old_log_prior_array,
    ) = get_weights_for_reweighting(
        result=result,
        old_likelihood=old_likelihood,
        new_likelihood=new_likelihood,
        old_prior=old_priors,
        new_prior=new_priors,
        npool=npool,
        resume_file=resume_file,
    )
    assert np.array_equal(
        old_log_likelihood_array, -np.ones(10) * 5 * np.log(2 * np.pi)
    )
    assert np.array_equal(
        new_log_likelihood_array, -np.ones(10) * 5 * np.log(2 * np.pi * 4)
    )
    assert np.array_equal(old_log_prior_array, -np.ones(10) * np.log(1))
    assert np.array_equal(new_log_prior_array, -np.ones(10) * np.log(2))
    assert np.array_equal(
        ln_weights,
        np.ones(10)
        * (5 * np.log(2 * np.pi) - 5 * np.log(2 * np.pi * 4) + np.log(1) - np.log(2)),
    )
    assert not os.path.exists(resume_file)
