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

>>> import numpy as np
>>> from regmodsm.model import Model
>>> model = Model(
        model_type="binomial",
        obs="obs_rate",
        dims=[{"name": "age_mid", "type": "numerical"}],
        var_builders=[{"col": "mean_BMI", "dim": "age_mid", "lam": 1.0, "uprior": (0.0, np.inf)}],
        weights="sample_size"
    )

Fit a RegMod model with intercept values smoothed by region. In this
model, a different intercept is fit for each unique value of
`region_id`. A Gaussian prior with mean 0 and standard deviation
1/sqrt(lam) is set on the mean of the intercept values.

>>> from regmodsm.model import Model
>>> model = Model(
        model_type="binomial",
        obs="obs_rate",
        dims=[{"name": "region_id", "type": "categorical"}],
        var_builders=[{"col": "intercept", "dim": "region_id", "lam_mean": 1.0}],
        weights="sample_size"
    )

Fit a RegMod model with intercept values smoothed by age-year. In this
model, a different intercept is fit for each unique age_group_id-year_id
pair.

>>> from regmodsm.model import Model
>>> model = Model(
        model_type="binomial",
        obs="obs_rate",
        dims=[{"name": ["age_group_id", "year_id"], "type": 2*["categorical"]}],
        var_builders=[{"col": "intercept", "dim": "age_group_id*year_id"}],
        "weights"="sample_size"
    )

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.stats import norm

from regmodsm.linalg import get_pred_var
from regmodsm.space import Space
from regmodsm.variable_builder import VariableBuilder
from regmodsm.regmod_builder import build_regmod_model
from regmodsm._typing import DataFrame, NDArray, RegmodModel


class Model:
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
            param_specs=param_specs or {},
        )
        self.spaces = spaces
        self.var_builders = var_builders

        self.core: RegmodModel | None = None

    @classmethod
    def from_config(cls, config: dict) -> Model:
        spaces = list(map(Space.from_config, config["spaces"]))
        var_builder_from_config = lambda config: VariableBuilder.from_config(
            config, spaces={space.name: space for space in spaces}
        )
        var_builders = list(map(var_builder_from_config, config["var_builders"]))

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
        mat, sd = block_diag(*mat), np.hstack(sd)
        return [dict(mat=mat, mean=0.0, sd=sd)]

    def _build_core(self) -> RegmodModel:
        return build_regmod_model(**self.core_config)

    def _encode(self, data: DataFrame) -> DataFrame:
        data = data.copy()
        data["intercept"] = 1.0

        for var_builder in self.var_builders:
            data = pd.concat([data, var_builder.encode(data)], axis=1)

        return data

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
        self._set_core_config(data_span or data)
        self.core = self._build_core()
        data = self._encode(data)
        self.core.attach_df(data)
        self.core.fit(**optimizer_options)
        _detach_df(self.core)

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
        data = self._encode(data)
        self.core.data.attach_df(data)
        param = self.core.params[0]
        coef = self.core.opt_coefs

        offset = np.zeros(len(data))
        if param.offset is not None:
            offset = data[param.offset].to_numpy()

        mat = np.ascontiguousarray(param.get_mat(self.core.data))

        lin_param = offset + mat.dot(coef)
        pred = param.inv_link.fun(lin_param)

        if return_ui:
            if alpha < 0 or alpha > 0.5:
                raise ValueError("`alpha` has to be between 0 and 0.5")
            vcov = self.core.opt_vcov
            lin_param_sd = np.sqrt(get_pred_var(mat, vcov))
            lin_param_lower = norm.ppf(0.5 * alpha, loc=lin_param, scale=lin_param_sd)
            lin_param_upper = norm.ppf(
                1 - 0.5 * alpha, loc=lin_param, scale=lin_param_sd
            )
            pred = np.vstack(
                [
                    pred,
                    param.inv_link.fun(lin_param_lower),
                    param.inv_link.fun(lin_param_upper),
                ]
            )
        self.core.data.detach_df()
        return pred


def _detach_df(model: RegmodModel) -> RegmodModel:
    """Detach input data and model arrays from the RegMod model."""
    model.data.detach_df()
    del model.mat
    del model.uvec
    del model.gvec
    del model.linear_uvec
    del model.linear_gvec
    del model.linear_umat
    del model.linear_gmat

    return model
