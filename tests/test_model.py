import numpy as np
import pandas as pd
import pytest

from spxmod.model import XModel


@pytest.fixture
def data() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "age": rng.choice([1.0, 2.0, 3.0, 4.0], n),
            "loc": rng.choice([1, 2, 3], n),
            "x": rng.normal(size=n),
            "weight": 20.0,
        }
    )
    lin = 0.2 * df["age"] + 0.5 * df["x"]
    df["obs_binomial"] = rng.binomial(20, 1 / (1 + np.exp(-lin))) / 20
    df["obs_poisson"] = rng.poisson(np.exp(lin)).astype(float)
    df["obs_gaussian"] = lin + rng.normal(scale=0.1, size=n)
    return df


def make_config(model_type: str) -> dict:
    return {
        "model_type": model_type,
        "obs": f"obs_{model_type}",
        "spaces": [
            {"dims": [{"name": "age", "dim_type": "numerical"}]},
            {"dims": [{"name": "loc", "dim_type": "categorical"}]},
        ],
        "var_builders": [
            {"name": "intercept", "space": "age", "lam": 1.0, "lam_mean": 1.0},
            {"name": "x", "space": "loc", "lam": 1.0},
            {"name": "x"},
        ],
    }


@pytest.mark.parametrize("model_type", ["binomial", "poisson", "gaussian"])
def test_fit_predict(data, model_type):
    model = XModel.from_config(make_config(model_type))
    model.fit(data)
    pred = model.predict(data)

    # 4 age cells + 3 loc cells + 1 plain covariate
    assert model.core.opt_coefs.shape == (8,)
    assert pred.shape == (len(data),)
    assert np.all(np.isfinite(pred))
    if model_type == "binomial":
        assert np.all((pred > 0) & (pred < 1))
    if model_type == "poisson":
        assert np.all(pred > 0)


def test_from_config_builds_objects(data):
    model = XModel.from_config(make_config("gaussian"))
    assert [space.name for space in model.spaces] == ["age", "loc"]
    assert [vb.name for vb in model.var_builders] == ["intercept", "x", "x"]
    assert model.var_builders[1].space is model.spaces[1]
    assert model.var_builders[2].space.size == 1


def test_fit_twice_does_not_duplicate_priors(data):
    model = XModel.from_config(make_config("gaussian"))
    model.fit(data)
    n_gpriors = len(model.core_config["linear_gpriors"])
    n_upriors = len(model.core_config["linear_upriors"])
    coefs = model.core.opt_coefs.copy()

    model.fit(data)
    assert len(model.core_config["linear_gpriors"]) == n_gpriors
    assert len(model.core_config["linear_upriors"]) == n_upriors
    assert np.allclose(model.core.opt_coefs, coefs)


def test_data_span(data):
    span = pd.DataFrame(
        {"age": [1.0, 2.0, 3.0, 4.0, 5.0], "loc": [1, 2, 3, 4, 5]}
    )
    model = XModel.from_config(make_config("gaussian"))
    model.fit(data, data_span=span)
    # 5 age cells + 5 loc cells + 1
    assert model.core.opt_coefs.shape == (11,)


def test_density(data):
    config = make_config("gaussian")
    config["var_builders"] = [
        {"name": "intercept", "space": "age", "lam": 1.0, "lam_mean": 1.0}
    ]
    config["spaces"] = config["spaces"][:1]
    density = pd.Series(
        [1.0, 2.0, 3.0, 4.0], index=pd.Index([1.0, 2.0, 3.0, 4.0], name="age")
    )
    model = XModel.from_config(config)
    model.fit(data, density={("intercept", "age"): density})
    pred = model.predict(data, density={("intercept", "age"): density})
    assert np.all(np.isfinite(pred))


def test_predict_return_ui(data):
    model = XModel.from_config(make_config("binomial"))
    model.fit(data)
    pred = model.predict(data, return_ui=True, alpha=0.1)
    assert pred.shape == (3, len(data))
    assert np.all(pred[1] <= pred[0]) and np.all(pred[0] <= pred[2])
    assert np.allclose(pred[0], model.predict(data))


def test_predict_before_fit(data):
    model = XModel.from_config(make_config("gaussian"))
    with pytest.raises(AttributeError):
        model.predict(data)


def test_param_specs(data):
    config = make_config("poisson")
    config["param_specs"] = {"offset": "x"}
    model = XModel.from_config(config)
    model.fit(data)
    assert model.core.params[0].offset == "x"
