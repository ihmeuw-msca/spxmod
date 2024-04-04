import numpy as np
import pandas as pd
import pytest

from regmodsm.model import Dimension, VarGroup


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
def dimensions(data) -> dict[str, Dimension]:
    dimensions = {
        "age": Dimension(name="age", type="numerical"),
        "loc": Dimension(name="loc", type="categorical"),
    }
    for dim in dimensions.values():
        dim.set_span(data=data)
    return dimensions


@pytest.mark.xfail(
    reason="lam for multi-indexed dimensions need to be addressed in another PR"
)
@pytest.mark.parametrize(("lam", "gprior_sd"), [(0.0, np.inf), (1.0, 1.0)])
def test_categorical_lam(dimensions, lam, gprior_sd):
    dim = dimensions["loc"]
    var_group = VarGroup(col="intercept", dim=dim, lam=lam)
    assert var_group.gprior.sd == gprior_sd


@pytest.mark.xfail(
    reason="lam for multi-indexed dimensions need to be addressed in another PR"
)
@pytest.mark.parametrize(
    ("lam", "scale_by_distance", "smooth_gprior_sd"),
    [
        (0.0, False, np.array([1.0 / np.sqrt(1e-8)])),
        (0.0, True, np.array([1.0 / np.sqrt(1e-8)])),
        (1.0, False, np.array([1.0, 1.0, 1.0 / np.sqrt(1e-8)])),
        (1.0, True, np.array([1.0, 3.0, 1.0 / np.sqrt(1e-8)])),
    ],
)
def test_numerical_lam(dimensions, lam, scale_by_distance, smooth_gprior_sd):
    dim = dimensions["age"]
    var_group = VarGroup(
        col="sdi", dim=dim, lam=lam, scale_by_distance=scale_by_distance
    )
    _, vec = var_group.get_smoothing_gprior()
    assert np.allclose(vec[1], smooth_gprior_sd)


def test_expand_data(data, dimensions):
    dim = dimensions["age"]
    var_group = VarGroup(col="sdi", dim=dim, lam=1.0)
    expanded_data = var_group.expand_data(data)
    assert expanded_data.shape == (6, 3)
    assert np.allclose(expanded_data["sdi_age_0"], [1, 0, 0, 1, 0, 0])
    assert np.allclose(expanded_data["sdi_age_1"], [0, 0, 2, 0, 0, 2])
    assert np.allclose(expanded_data["sdi_age_2"], [0, 3, 0, 0, 3, 0])
