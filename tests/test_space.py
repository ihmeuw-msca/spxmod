import numpy as np
import pandas as pd
import pytest

from regmodsm.space import Space


@pytest.fixture
def space() -> Space:
    return Space(
        dims=[
            dict(name="location_id", dim_type="categorical"),
            dict(name="age_mid", dim_type="numerical"),
        ]
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
            dict(
                location_id=[1, 1, 1, 2, 2, 2],
                age_mid=[1, 1.5, 3, 1, 1.5, 3],
            )
        )
    )


def test_encode(space, data):
    space.set_span(data=data)
    variables = space.encode(data=data, column="sdi")
    variables = variables.astype(float)
    assert variables.equals(
        pd.DataFrame(
            {
                "sdi_location_id*age_mid_0": [1.0, 0, 0, 0, 0],
                "sdi_location_id*age_mid_1": [0, 2.0, 0, 0, 0],
                "sdi_location_id*age_mid_2": [0, 0, 3.0, 0, 0],
                "sdi_location_id*age_mid_3": [0, 0, 0, 4.0, 0],
                "sdi_location_id*age_mid_4": [0, 0, 0, 0, 5.0],
                "sdi_location_id*age_mid_5": [0, 0, 0, 0, 0.0],
            }
        )
    )


def test_build_smoothing_prior(space, data):
    space.set_span(data=data)
    prior = space.build_smoothing_prior(lam=1.0, lam_mean=0.0)

    assert np.allclose(
        prior["mat"],
        np.array(
            [
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            ]
        ),
    )

    assert np.allclose(prior["sd"], np.array([1.0, 1.0, 1.0, 1.0]))
