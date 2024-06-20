"""Fit a RegMod model with variables regularized in dimension groupings.

Examples
--------
Fit a RegMod model with covariate values smoothed by age. In this model,
a different coefficient for the covariate `mean_BMI` is fit for each
unique value of `age_mid`. Because the age dimension is numerical, a
Gaussian prior with mean 0 and standard deviation 1/sqrt(lam) is set on
the differences between neighboring coefficients. This ensures that
coefficients vary smoothly by age. The uniform prior forces the
coefficients to have positive values.

>>> from spxmod.model import XModel
>>> config = dict(
        model_type="binomial",
        obs="obs_rate",
        spaces=[dict(dims=[{"name": "age_mid", "dim_type": "numerical"}])],
        var_builders=[dict(name="mean_BMI", space="age_mid", lam=1.0, uprior=dict(lb=0.0))],
        weights="sample_size",
    )
>>> model = XModel.from_config(config)

Fit a RegMod model with intercept values smoothed by region. In this
model, a different intercept is fit for each unique value of
`region_id`. A Gaussian prior with mean 0 and standard deviation
1/sqrt(lam) is set on the mean of the intercept values.

>>> from spxmod.model import XModel
>>> config = dict(
        model_type="binomial",
        obs="obs_rate",
        spaces=[dict(dims=[dict(name="region_id", dim_type="categorical")])],
        var_builders=[dict(name="intercept", space="region_id", lam_mean=1.0)],
        weights="sample_size",
    )
>>> model = XModel.from_config(config)

Fit a RegMod model with intercept values smoothed by age-year. In this
model, a different intercept is fit for each unique age_group_id-year_id
pair.

>>> from spxmod.model import XModel
>>> config = dict(
        model_type="binomial",
        obs="obs_rate",
        spaces=[
            dict(
                dims=[
                    dict(name="age_group_id", dim_type="categorical"),
                    dict(name="year_id", dim_type="categorical"),
                ],
            ),
        ],
        var_builders=[dict(name="intercept", space="age_group_id*year_id")],
        weights="sample_size",
    )
>>> model = XModel.from_config(config)

"""

from __future__ import annotations

import functools

import numpy as np
from scipy.sparse import block_diag, coo_matrix, hstack

from spxmod.regmod_builder import build_regmod_model
from spxmod.space import Space
from spxmod.typing import DataFrame, NDArray, RegmodModel
from spxmod.variable_builder import VariableBuilder


class XModel:
    """RegMod Smooth model.

    Parameters
    ----------
    model_type : {"binomial", "poisson", "gaussian"}
        RegMod model type.
    obs : str
        Name of the observation column in the data.
    spaces : list[dict]
        List of dictionaries containing space names and arguments.
    var_builders : list[dict]
        List of dictionaries containing variable group names and arguments.
    weights : str, optional
        Name of the weight column in the data. Default is "weight".
    param_specs : dict, optional
        Additional parameter specifications for the model.

    """

    def __init__(
        self,
        model_type: str,
        obs: str,
        spaces: list[Space],
        var_builders: list[VariableBuilder],
        weights: str = "weight",
        param_specs: dict | None = None,
    ) -> None:
        self.core_config = dict(
            model_type=model_type,
            data=dict(col_obs=obs, col_weights=weights),
            variables=[],
            linear_gpriors=[],
        )
        if param_specs is not None:
            self.core_config.update(param_specs)
        self.spaces = spaces
        self.var_builders = var_builders

        self.core: RegmodModel | None = None

    @classmethod
    def from_config(cls, config: dict) -> XModel:
        spaces = list(map(Space.from_config, config["spaces"]))
        var_builder_from_config = functools.partial(
            VariableBuilder.from_config,
            spaces={space.name: space for space in spaces},
        )
        var_builders = list(
            map(var_builder_from_config, config["var_builders"])
        )

        config["spaces"] = spaces
        config["var_builders"] = var_builders
        return cls(**config)

    def _set_core_config(self, data: DataFrame) -> None:
        for space in self.spaces:
            space.set_span(data)

        self.core_config["variables"] = self._build_variables()
        self.core_config["linear_gpriors"] = self._build_linear_gpriors()

    def _build_variables(self) -> list[dict]:
        variables = []
        for var_builder in self.var_builders:
            variables.extend(var_builder.build_variables())
        return variables

    def _build_linear_gpriors(self) -> list[dict]:
        mat, sd = [], []
        for var_builder in self.var_builders:
            prior = var_builder.build_smoothing_prior()
            mat.append(prior["mat"]), sd.append(prior["sd"])
        mat, sd = block_diag(mat), np.hstack(sd)
        return [dict(mat=mat, mean=0.0, sd=sd)]

    def _build_core(self) -> RegmodModel:
        return build_regmod_model(**self.core_config)

    def _encode(self, data: DataFrame) -> coo_matrix:
        mats = [var_builder.encode(data) for var_builder in self.var_builders]
        return hstack(mats)

    def fit(
        self,
        data: DataFrame,
        data_span: DataFrame | None = None,
        **optimizer_options,
    ) -> None:
        """Fit the model to the data.

        Parameters
        ----------
        data : DataFrame
            Data to fit the model to.
        data_span : DataFrame, optional
            Data containing the unique dimension values. If None, values
            are extracted from the data. Default is None.
        optimizer_options
            Additional options for the optimizer.

        """
        self._set_core_config(data if data_span is None else data_span)
        self.core = self._build_core()
        self.core.fit(data, self._encode, **optimizer_options)

    def predict(
        self,
        data: DataFrame,
        return_ui: bool = False,
        alpha: float = 0.05,
    ) -> NDArray:
        """Predict the response variable.

        Parameters
        ----------
        data : DataFrame
            Data to predict the response variable.
        return_ui : bool, optional
            Whether to return the prediction interval. Default is False.
        alpha : float, optional
            Significance level for the prediction interval. Default is 0.05.

        Returns
        -------
        NDArray
            Predicted response variable. If `return_ui` is True, the prediction interval
            is also returned.

        """
        return self.core.predict(data, self._encode, return_ui, alpha)
