import numpy as np
import pandas as pd
import pytest

from regmodsm.dimension import Dimension


@pytest.fixture
def dim() -> Dimension:
    return Dimension(name=["location_id", "age_mid"], type=["categorical", "numerical"])


@pytest.fixture
def data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "location_id": [1, 1, 1, 2, 2],
            "age_mid": [1, 1.5, 3, 1, 1.5],
            "sdi": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def test_set_span(dim, data):
    dim.set_span(data=data)
    assert dim.span.equals(
        pd.DataFrame(
            dict(
                index=[0, 1, 2, 3, 4, 5],
                location_id=[1, 1, 1, 2, 2, 2],
                age_mid=[1, 1.5, 3, 1, 1.5, 3],
            )
        )
    )


def test_get_dummy_names(dim, data):
    dim.set_span(data=data)
    assert dim.label == "location_id*age_mid"
    dummy_names = dim.get_dummy_names(column="sdi")
    assert dummy_names == [
        "sdi_location_id*age_mid_0",
        "sdi_location_id*age_mid_1",
        "sdi_location_id*age_mid_2",
        "sdi_location_id*age_mid_3",
        "sdi_location_id*age_mid_4",
        "sdi_location_id*age_mid_5",
    ]


def test_get_dummies(dim, data):
    dim.set_span(data=data)
    dummies = dim.get_dummies(data=data, column="sdi")
    dummies = dummies.astype(float)
    assert dummies.equals(
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


def test_get_smoothing_gprior(dim, data):
    dim.set_span(data=data)
    mat, vec = dim.get_smoothing_gprior(lam=1.0, lam_mean=0.0)

    assert np.allclose(
        mat,
        np.array(
            [
                [1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, -1.0],
            ]
        ),
    )

    assert np.allclose(vec, np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]))
