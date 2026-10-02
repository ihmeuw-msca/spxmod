import numpy as np
import pandas as pd
import pytest

from spxmod.dimension import (
    CategoricalDimension,
    NumericalDimension,
    build_dimension,
)


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [1.0, 3.0, 2.0, 1.0, np.nan],
            "age_lb": [0.5, 2.5, 1.5, 0.5, np.nan],
            "age_ub": [1.5, 3.5, 2.5, 1.5, np.nan],
            "loc": [1, 2, 2, 1, 1],
        }
    )


def test_build_dimension():
    assert isinstance(
        build_dimension("loc", "categorical"), CategoricalDimension
    )
    assert isinstance(build_dimension("age", "numerical"), NumericalDimension)
    with pytest.raises(TypeError):
        build_dimension("age", "unknown")


def test_categorical_rejects_interval():
    with pytest.raises(ValueError):
        CategoricalDimension("age", interval=("age_lb", "age_ub"))


def test_span_not_set():
    dim = NumericalDimension("age")
    with pytest.raises(ValueError):
        _ = dim.span
    with pytest.raises(ValueError):
        _ = dim.grid


def test_set_span_point(data):
    dim = NumericalDimension("age")
    dim.set_span(data)
    assert np.allclose(dim.span, [1.0, 2.0, 3.0])
    assert np.allclose(dim.grid, dim.span)
    assert dim.size == 3


def test_set_span_keepna(data):
    dim = CategoricalDimension("age", skipna=False)
    dim.set_span(data)
    assert dim.size == 4
    assert np.isnan(dim.span).sum() == 1


def test_set_span_interval(data):
    dim = NumericalDimension("age", interval=("age_lb", "age_ub"))
    dim.set_span(data)
    assert np.allclose(dim.grid, [0.5, 1.5, 2.5, 3.5])
    assert np.allclose(dim.span, [1.0, 2.0, 3.0])
    assert dim.size == 3


def test_set_span_interval_gap():
    data = pd.DataFrame({"lb": [0.0, 2.0], "ub": [1.0, 3.0]})
    dim = NumericalDimension("x", interval=("lb", "ub"))
    with pytest.raises(ValueError, match="gap"):
        dim.set_span(data)


def test_encode_coords_point(data):
    dim = NumericalDimension("age")
    dim.set_span(data)
    coords = pd.DataFrame({"age": [3.0, 1.0, 2.0]})
    weights = dim.encode_coords(coords)
    assert weights["row"].tolist() == [0, 1, 2]
    assert weights["age_col"].tolist() == [2, 0, 1]
    assert np.allclose(weights["age_val"], 1.0)


def test_encode_coords_interval(data):
    dim = NumericalDimension("age", interval=("age_lb", "age_ub"))
    dim.set_span(data)
    # row 0 covers cells 0 and 1 fully, row 1 covers half of cell 1
    coords = pd.DataFrame({"age_lb": [0.5, 2.0], "age_ub": [2.5, 2.5]})
    weights = dim.encode_coords(coords)
    assert weights["row"].tolist() == [0, 0, 1]
    assert weights["age_col"].tolist() == [0, 1, 1]
    assert np.allclose(weights["age_val"], [1.0, 1.0, 0.5])


def test_build_smoothing_mat(data):
    dim = NumericalDimension("age")
    dim.set_span(data)
    mat = dim.build_smoothing_mat().toarray()
    assert np.allclose(mat, [[1, -1, 0], [0, 1, -1]])


@pytest.mark.parametrize(
    ("scale_by_distance", "expected"),
    [(False, [0.5, 0.5]), (True, [0.5, 1.0])],
)
def test_build_smoothing_sd(scale_by_distance, expected):
    dim = NumericalDimension("age")
    dim.set_span(pd.DataFrame({"age": [1.0, 2.0, 4.0]}))
    sd = dim.build_smoothing_sd(lam=4.0, scale_by_distance=scale_by_distance)
    assert np.allclose(sd, expected)


def test_build_order_mat(data):
    dim = NumericalDimension("age")
    dim.set_span(data)
    mat = dim.build_order_mat([2, 0]).toarray()
    assert np.allclose(mat, [[-1, 0, 1]])
