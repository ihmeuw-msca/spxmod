import numpy as np
import pandas as pd
import pytest

from regmodsm.space import Space
from regmodsm.vargroup import VarGroup


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "intercept": [1, 1, 1, 1, 1, 1],
            "sdi": [1, 3, 2, 1, 3, 2],
            "loc": [1, 1, 1, 2, 2, 2],
            "age": [1, 3, 1.5, 1, 3, 1.5],
        }
    )


@pytest.fixture
def dimensions() -> dict[str, dict]:
    return {
        "age": dict(name="age", dim_type="numerical"),
        "loc": dict(name="loc", dim_type="categorical"),
    }


@pytest.mark.parametrize(("lam", "gprior_sd"), [(0.0, np.inf), (1.0, 1.0)])
def test_categorical_lam(data, dimensions, lam, gprior_sd):
    space = Space(dims=[dimensions["loc"]])
    space.set_span(data)
    var_group = VarGroup(name="intercept", space=space, lam=lam)
    assert var_group.gprior.sd == gprior_sd


@pytest.mark.parametrize(
    ("lam", "scale_by_distance", "smooth_gprior_sd"),
    [
        (0.0, False, np.array([])),
        (0.0, True, np.array([])),
        (1.0, False, np.array([1.0, 1.0])),
        (1.0, True, np.array([1.0, 3.0])),
    ],
)
def test_numerical_lam(data, dimensions, lam, scale_by_distance, smooth_gprior_sd):
    space = Space(dims=[dimensions["age"]])
    space.set_span(data)
    var_group = VarGroup(
        name="sdi", space=space, lam=lam, scale_by_distance=scale_by_distance
    )
    prior = var_group.create_smoothing_prior()
    assert np.allclose(prior["sd"], smooth_gprior_sd)


def test_encode(data, dimensions):
    space = Space(dims=[dimensions["age"]])
    space.set_span(data)
    var_group = VarGroup(name="sdi", space=space, lam=1.0)
    expanded_data = var_group.encode(data)
    assert expanded_data.shape == (6, 3)
    assert np.allclose(expanded_data["sdi_age_0"], [1, 0, 0, 1, 0, 0])
    assert np.allclose(expanded_data["sdi_age_1"], [0, 0, 2, 0, 0, 2])
    assert np.allclose(expanded_data["sdi_age_2"], [0, 3, 0, 0, 3, 0])
