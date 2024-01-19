import numpy as np
import pandas as pd
import pytest

from regmodsm.model import Dimension, VarGroup


@pytest.fixture
def dimensions() -> dict[str, Dimension]:
    df = pd.DataFrame(
        {
            "loc": [1, 1, 1, 2, 2, 2],
            "age": [1, 1.5, 3, 1, 1.5, 3],
        }
    )
    dimensions = {
        "age": Dimension(name="age", type="continuous"),
        "loc": Dimension(name="loc", type="categorical"),
    }
    for dim in dimensions.values():
        dim.set_vals(df)
    return dimensions


@pytest.mark.parametrize(("lam", "gprior_sd"), [(0.0, np.inf), (1.0, 1.0)])
def test_categorical_lam(dimensions, lam, gprior_sd):
    dim = dimensions["loc"]
    var_group = VarGroup(col="loc", dim=dim, lam=lam)
    assert var_group.gprior.sd == gprior_sd


@pytest.mark.parametrize(
    ("lam", "scale_by_distance", "smooth_gprior_sd"),
    [
        (0.0, False, np.array([1.0 / np.sqrt(1e-8)])),
        (0.0, True, np.array([1.0 / np.sqrt(1e-8)])),
        (1.0, False, np.array([1.0, 1.0, 1.0 / np.sqrt(1e-8)])),
        (1.0, True, np.array([1.0, 3.0, 1.0 / np.sqrt(1e-8)])),
    ],
)
def test_continuous_lam(dimensions, lam, scale_by_distance, smooth_gprior_sd):
    dim = dimensions["age"]
    var_group = VarGroup(
        col="age", dim=dim, lam=lam, scale_by_distance=scale_by_distance
    )
    _, vec = var_group.get_smoothing_gprior()
    assert np.allclose(vec[1], smooth_gprior_sd)
