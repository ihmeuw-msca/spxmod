from typing import Callable

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sps
from regmod.prior import LinearUniformPrior

from spxmod.regmod_builder import (
    BinomialModel,
    Data,
    SparseBinomialModel,
    Variable,
    get_vcov,
)


@pytest.fixture
def sim_data() -> pd.DataFrame:
    np.random.seed(123)
    return pd.DataFrame(
        {
            "obs_rate": np.random.binomial(10, 0.5, size=20) / 10,
            "sample_size": 10.0,
            **{f"cov_{i}": np.random.rand(20) for i in range(10)},
        }
    )


@pytest.fixture
def data() -> Data:
    return Data(col_obs="obs_rate", col_weights="sample_size")


@pytest.fixture
def variables(sim_data) -> list[Variable]:
    var_names = [col for col in sim_data.columns if col.startswith("cov_")]
    return [Variable(var_name) for var_name in var_names]


@pytest.fixture
def linear_upriors(variables) -> tuple[LinearUniformPrior, LinearUniformPrior]:
    size = len(variables)
    mat = sps.coo_matrix(sps.vstack([sps.identity(size), np.zeros((1, size))]))
    lb, ub = -np.inf, 0.0
    ref_linear_uprior = LinearUniformPrior(mat=mat.toarray(), lb=lb, ub=ub)
    alt_linear_uprior = LinearUniformPrior(mat=mat, lb=lb, ub=ub)
    return ref_linear_uprior, alt_linear_uprior


@pytest.fixture
def ref_model(data, variables, linear_upriors) -> BinomialModel:
    return BinomialModel(
        data,
        param_specs={
            "p": {"variables": variables, "linear_upriors": [linear_upriors[0]]}
        },
    )


@pytest.fixture
def alt_model(data, variables, linear_upriors) -> SparseBinomialModel:
    data = dict(col_obs=data.col_obs, col_weights=data.col_weights)
    variables = [dict(name=v.name) for v in variables]
    linear_uprior = linear_upriors[1]
    linear_uprior = dict(
        mat=linear_uprior.mat, lb=linear_uprior.lb, ub=linear_uprior.ub
    )

    return SparseBinomialModel(data, variables, linear_upriors=[linear_uprior])


@pytest.fixture
def encode(variables) -> Callable:
    def _encode(df):
        return df[[v.name for v in variables]].to_numpy()

    return _encode


def test_parsing_constraints(sim_data, ref_model, alt_model, encode):
    ref_model.attach_df(sim_data)
    alt_model.attach_df(sim_data, encode=encode)

    assert np.allclose(ref_model.cmat, alt_model.cmat.toarray())
    assert np.allclose(ref_model.cvec, alt_model.cvec)


def test_model_fitting(sim_data, ref_model, alt_model, encode):
    ref_model.attach_df(sim_data)
    ref_model.fit()
    alt_model.fit(sim_data, encode)

    assert np.allclose(alt_model.opt_coefs, ref_model.opt_coefs)
    assert np.allclose(
        get_vcov(alt_model.opt_hessian, alt_model.opt_jacobian2),
        ref_model.opt_vcov,
    )
