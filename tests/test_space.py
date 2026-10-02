import numpy as np
import pandas as pd
import pytest

from spxmod.space import Space


@pytest.fixture
def space() -> Space:
    return Space.from_config(
        {
            "dims": [
                {"name": "location_id", "dim_type": "categorical"},
                {"name": "age_mid", "dim_type": "numerical"},
            ]
        }
    )


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": [1, 1, 1, 2, 2],
            "age_mid": [1, 1.5, 3, 1, 1.5],
            "sdi": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_set_span(space, data):
    space.set_span(data=data)
    assert space.span.equals(
        pd.DataFrame(
            {
                "location_id": [1, 1, 1, 2, 2, 2],
                "age_mid": [1, 1.5, 3, 1, 1.5, 3],
            }
        )
    )


def test_encode(space, data):
    space.set_span(data=data)
    mat = data[["sdi"]].to_numpy()
    coords = data[space.dim_names]
    mat = space.encode(mat, coords)
    assert np.allclose(mat.toarray(), np.diag(np.arange(1, 7, dtype=float))[:5])


def test_build_smoothing_prior(space, data):
    space.set_span(data=data)
    prior = space.build_smoothing_prior(size=2, lam=1.0, lam_mean=0.0)

    assert np.allclose(
        prior["mat"].toarray(),
        np.array(
            [
                [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0],
            ]
        ),
    )

    assert np.allclose(
        prior["sd"], np.repeat(np.array([1.0, 1.0, 1.0, 1.0]), 2)
    )


@pytest.fixture
def interval_space() -> Space:
    space = Space.from_config(
        {
            "dims": [
                {
                    "name": "age",
                    "dim_type": "numerical",
                    "interval": ("age_lb", "age_ub"),
                }
            ]
        }
    )
    space.set_span(pd.DataFrame({"age_lb": [0.0, 1.0], "age_ub": [1.0, 2.0]}))
    return space


def test_empty_space():
    space = Space()
    assert space.size == 1
    assert space.dims == []
    assert space.build_encoded_names("x") == ["x"]
    coords = pd.DataFrame(index=range(3))
    mat = space.encode(np.ones((3, 1)), coords).toarray()
    assert np.allclose(mat, np.ones((3, 1)))


def test_encode_interval_normalized(interval_space):
    # one row spanning both cells equally
    coords = pd.DataFrame({"age_lb": [0.0], "age_ub": [2.0]})
    mat = interval_space.encode(np.array([[2.0]]), coords).toarray()
    assert np.allclose(mat, [[1.0, 1.0]])


def test_encode_with_density(interval_space):
    coords = pd.DataFrame({"age_lb": [0.0], "age_ub": [2.0]})
    density = pd.Series([1.0, 3.0], index=pd.Index([0.5, 1.5], name="age"))
    mat = interval_space.encode(
        np.array([[1.0]]), coords, density=density
    ).toarray()
    assert np.allclose(mat, [[0.25, 0.75]])


def test_density_wrong_type(interval_space):
    coords = pd.DataFrame({"age_lb": [0.0], "age_ub": [2.0]})
    with pytest.raises(TypeError):
        interval_space.encode(np.ones((1, 1)), coords, density=[1.0, 3.0])


def test_density_missing_index(interval_space):
    coords = pd.DataFrame({"age_lb": [0.0], "age_ub": [2.0]})
    density = pd.Series([1.0, 3.0], index=pd.Index([0.5, 1.5], name="year"))
    with pytest.raises(ValueError, match="density index"):
        interval_space.encode(np.ones((1, 1)), coords, density=density)


def test_density_missing_value(interval_space):
    coords = pd.DataFrame({"age_lb": [0.0], "age_ub": [2.0]})
    density = pd.Series([1.0], index=pd.Index([0.5], name="age"))
    with pytest.raises(ValueError, match="Missing density"):
        interval_space.encode(np.ones((1, 1)), coords, density=density)


def test_build_smoothing_prior_lam_mean(space, data):
    space.set_span(data=data)
    prior = space.build_smoothing_prior(size=1, lam=0.0, lam_mean=4.0)
    assert prior["mat"].shape == (1, space.size)
    assert np.allclose(prior["mat"].toarray(), 1.0 / space.size)
    assert np.allclose(prior["sd"], [0.5])


def test_build_order_prior_empty(space, data):
    space.set_span(data=data)
    prior = space.build_order_prior()
    assert prior["mat"].shape == (0, space.size)


def test_build_order_prior(space, data):
    space.set_span(data=data)
    # age_mid has 3 cells, location_id has 2; order age cells 0 < 1 < 2
    prior = space.build_order_prior(order_dim="age_mid", order=[[0, 1, 2]])
    mat = prior["mat"].toarray()
    expected = np.kron(np.eye(2), [[1, -1, 0], [0, 1, -1]])
    assert np.allclose(mat, expected)
