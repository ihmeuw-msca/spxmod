import numpy as np
import pandas as pd
import pytest

from spxmod.space import Space
from spxmod.variable_builder import VariableBuilder


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
    space = Space.from_config(dict(dims=[dimensions["loc"]]))
    space.set_span(data)
    var_builder = VariableBuilder(name="intercept", space=space, lam=lam)
    assert var_builder.gprior["sd"] == gprior_sd


@pytest.mark.parametrize(
    ("lam", "scale_by_distance", "smooth_gprior_sd"),
    [
        (0.0, False, np.array([])),
        (0.0, True, np.array([])),
        (1.0, False, np.array([1.0, 1.0])),
        (1.0, True, np.array([1.0, 3.0])),
    ],
)
def test_numerical_lam(
    data, dimensions, lam, scale_by_distance, smooth_gprior_sd
):
    space = Space.from_config(dict(dims=[dimensions["age"]]))
    space.set_span(data)
    var_builder = VariableBuilder(
        name="sdi", space=space, lam=lam, scale_by_distance=scale_by_distance
    )
    prior = var_builder.build_smoothing_prior()
    assert np.allclose(prior["sd"], smooth_gprior_sd)


def test_encode(data, dimensions):
    space = Space.from_config(dict(dims=[dimensions["age"]]))
    space.set_span(data)
    var_builder = VariableBuilder(name="sdi", space=space, lam=1.0)
    mat = var_builder.encode(data).toarray()
    assert mat.shape == (6, 3)
    assert np.allclose(mat[:, 0], [1, 0, 0, 1, 0, 0])
    assert np.allclose(mat[:, 1], [0, 0, 2, 0, 0, 2])
    assert np.allclose(mat[:, 2], [0, 3, 0, 0, 3, 0])


def test_encode_spline_variable(data, dimensions):
    space = Space.from_config(dict(dims=[dimensions["loc"], dimensions["age"]]))
    space.set_span(data)
    spline = dict(knots=[1, 2, 3], degree=2)
    var_builder = VariableBuilder(
        name="sdi", space=space, lam=1.0, spline=spline
    )
    mat = var_builder.encode(data).toarray()

    assert mat.shape == (len(data), var_builder.size)

    spline_mat = var_builder.spline.design_mat(data["sdi"])
    position_mat = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
        ]
    )
    my_mat = np.array(
        [np.outer(x, y).ravel() for x, y in zip(position_mat, spline_mat)]
    )
    assert np.allclose(my_mat, mat)
