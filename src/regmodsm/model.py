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
        var_groups=[{"col": "mean_BMI", "dim": "age_mid", "lam": 1.0, "uprior": (0.0, np.inf)}],
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
        var_groups=[{"col": "intercept", "dim": "region_id", "lam_mean": 1.0}],
        weights="sample_size"
    )

"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from regmod.data import Data
from regmod.models import BinomialModel, GaussianModel, PoissonModel
from regmod.models import Model as RegmodModel
from regmod.prior import LinearGaussianPrior, GaussianPrior, Prior, UniformPrior
from regmod.variable import Variable
from scipy.linalg import block_diag
from scipy.stats import norm

from regmodsm.linalg import get_pred_var
from regmodsm.dimension import Dimension

_model_dict = {
    "binomial": BinomialModel,
    "gaussian": GaussianModel,
    "poisson": PoissonModel,
}


class VarGroup:
    """Variable group created by partitioning a variable along a dimension.

    Parameters
    ----------
    col : str
        Name of the variable column in the data.
    dim : Dimension, optional
        Dimension to partition the variable on. If None, the variable is
        not partitioned.
    lam : float, optional
        Regularization parameter for the coefficients in the variable
        group. Default is 0. If the dimension is numerical, a Gaussian
        prior with mean 0 and standard deviation 1/sqrt(lam) is set on
        the differences between neighboring coefficients along the
        dimension. If the dimension is categorical, a Gaussian prior
        with mean 0 and standard deviation 1/sqrt(lam) is set on the
        coefficients.
    lam_mean : float, optional
        Regularization parameter for the mean of the coefficients in the
        variable group. Default is 1e-8. A Gaussian prior with mean 0
        and standard deviation 1/sqrt(lam_mean) is set on the mean of
        the coefficients.
    gprior : tuple, optional
        Gaussian prior for the variable. Default is (0, np.inf).
        Argument is overwritten with (0, 1/sqrt(lam)) if dimension is
        categorical.
    uprior : tuple, optional
        Uniform prior for the variable. Default is (-np.inf, np.inf).
    scale_by_distance : bool, optional
        Whether to scale the prior standard deviation by the distance
        between the neighboring values along the dimension. For
        numerical dimensions only. Default is False.

    """

    def __init__(
        self,
        col: str,
        dim: Dimension | None = None,
        lam: float = 0.0,
        lam_mean: float = 1e-8,
        gprior: tuple[float, float] = (0.0, np.inf),
        uprior: tuple[float, float] = (-np.inf, np.inf),
        scale_by_distance: bool = False,
    ) -> None:
        self.col = col
        self.dim = dim
        self.lam = lam
        self.lam_mean = lam_mean
        self._gprior = gprior
        self._uprior = uprior
        self.scale_by_distance = scale_by_distance

        # transfer lam to gprior when dim is categorical
        if self.dim is not None:
            if self.dim.type == "categorical" and self.lam > 0.0:
                self._gprior = (0.0, 1.0 / np.sqrt(self.lam))

    @property
    def gprior(self) -> GaussianPrior:
        """Gaussian prior for the variable group."""
        return GaussianPrior(mean=self._gprior[0], sd=self._gprior[1])

    @property
    def uprior(self) -> UniformPrior:
        """Uniform prior for the variable group."""
        return UniformPrior(lb=self._uprior[0], ub=self._uprior[1])

    @property
    def priors(self) -> list[Prior]:
        """List of Gaussian and Uniform priors for the variable group."""
        return [self.gprior, self.uprior]

    @property
    def size(self) -> int:
        """Number of variables in the variable group."""
        if self.dim is None:
            return 1
        return self.dim.size

    def get_variables(self) -> list[Variable]:
        """Returns the list of variables in the variable group."""
        if self.dim is None:
            return [Variable(self.col, priors=self.priors)]
        variables = [
            Variable(name, priors=self.priors)
            for name in self.dim.get_dummy_names(self.col)
        ]
        return variables

    def get_smoothing_gprior(self) -> tuple[NDArray, NDArray]:
        """Returns the smoothing Gaussian prior for the variable group.

        If the dimension is numerical and lam > 0, a Gaussian prior with
        mean 0 and standard deviation 1/sqrt(lam) is set on the
        differences between neighboring coefficients along the
        dimension. For both dimension types if lam_mean > 0, a Gaussian
        prior with mean 0 and standard deviation 1/sqrt(lam_mean) is set
        on the mean of the coefficients.

        Returns
        -------
        tuple[NDArray, NDArray]
            Smoothing Gaussian prior matrix and vector.

        """
        n = self.size
        mat = np.empty(shape=(0, n))
        vec = np.empty(shape=(2, 0))

        if self.dim is not None:
            if self.dim.type == "numerical" and self.lam > 0.0:
                mat = np.zeros(shape=(n - 1, n))
                id0 = np.diag_indices(n - 1)
                id1 = (id0[0], id0[1] + 1)
                mat[id0], mat[id1] = -1.0, 1.0
                vec = np.zeros(shape=(2, n - 1))
                vec[1] = 1 / np.sqrt(self.lam)
                if self.scale_by_distance:
                    delta = np.diff(self.dim.vals)
                    delta /= delta.min()
                    vec[1] *= delta

            if self.lam_mean > 0.0:
                mat = np.vstack([mat, np.repeat(1.0 / n, n)])
                vec = np.hstack(
                    [vec, np.array([[0.0], [1.0 / np.sqrt(self.lam_mean)]])]
                )

        return mat, vec

    def expand_data(self, data: DataFrame) -> DataFrame:
        """Expand the variable into multiple columns based on the dimension.

        Parameters
        ----------
        data : DataFrame
            Data containing the variable column.

        Returns
        -------
        DataFrame
            Expanded variable columns.

        """
        if self.dim is None:
            return DataFrame(index=data.index)
        return self.dim.get_dummies(data, column=self.col)


class Model:
    """RegMod Smooth model.

    Parameters
    ----------
    model_type : {"binomial", "poisson", "gaussian"}
        RegMod model type.
    obs : str
        Name of the observation column in the data.
    dims : list[dict]
        List of dictionaries containing dimension names and arguments.
    var_groups : list[dict]
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
        dims: list[dict],
        var_groups: list[dict],
        weights: str = "weight",
        param_specs: dict | None = None,
    ) -> None:
        self.model_type = model_type
        self.obs = obs

        self.dims = tuple(map(lambda kwargs: Dimension(**kwargs), dims))
        self._dim_dict = {dim.label: dim for dim in self.dims}

        for var_group in var_groups:
            if ("dim" in var_group) and (var_group["dim"] is not None):
                var_group["dim"] = self._dim_dict[var_group["dim"]]
        self.var_groups = tuple(map(lambda kwargs: VarGroup(**kwargs), var_groups))

        self.weights = weights
        self.param_specs = param_specs or {}

        self._model: RegmodModel | None = None

    def _set_dim_vals(self, data: DataFrame) -> None:
        for dim in self.dims:
            dim.set_vals(data)

    def _get_smoothing_prior(self) -> tuple[NDArray, NDArray]:
        prior_mats = []
        prior_vecs = []
        for var_group in self.var_groups:
            mat, vec = var_group.get_smoothing_gprior()
            prior_mats.append(mat)
            prior_vecs.append(vec)
        prior_mat = block_diag(*prior_mats)
        prior_vec = np.hstack(prior_vecs)
        return prior_mat, prior_vec

    def _get_model(self) -> RegmodModel:
        data = Data(
            col_obs=self.obs,
            col_weights=self.weights,
        )

        variables = []
        for var_group in self.var_groups:
            variables.extend(var_group.get_variables())

        linear_gpriors = []
        prior_mat, prior_vec = self._get_smoothing_prior()
        if len(prior_mat) > 0:
            linear_gpriors = [
                LinearGaussianPrior(mat=prior_mat, mean=prior_vec[0], sd=prior_vec[1])
            ]

        model_class = _model_dict[self.model_type]
        model_param = model_class.param_names[0]

        model = model_class(
            data,
            param_specs={
                model_param: {
                    "variables": variables,
                    "linear_gpriors": linear_gpriors,
                    **self.param_specs,
                }
            },
        )
        return model

    def _expand_data(self, data: DataFrame) -> DataFrame:
        data = data.copy()
        data["intercept"] = 1.0

        for var_group in self.var_groups:
            df_covs = var_group.expand_data(data)
            data = pd.concat([data, df_covs], axis=1)

        return data

    def fit(
        self,
        data: DataFrame,
        data_dim_vals: DataFrame | None = None,
        **optimizer_options,
    ) -> None:
        """Fit the model to the data.

        Parameters
        ----------
        data : DataFrame
            Data to fit the model to.
        data_dim_vals : DataFrame, optional
            Data containing the unique dimension values. If None, values
            are extracted from the data. Default is None.
        optimizer_options
            Additional options for the optimizer.

        """
        if data_dim_vals is None:
            self._set_dim_vals(data)
        else:
            self._set_dim_vals(data_dim_vals)
        self._model = self._get_model()
        data = self._expand_data(data)
        self._model.attach_df(data)
        self._model.fit(**optimizer_options)
        _detach_df(self._model)

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
        data = self._expand_data(data)
        self._model.data.attach_df(data)
        param = self._model.params[0]
        coef = self._model.opt_coefs

        offset = np.zeros(len(data))
        if param.offset is not None:
            offset = data[param.offset].to_numpy()

        mat = np.ascontiguousarray(param.get_mat(self._model.data))

        lin_param = offset + mat.dot(coef)
        pred = param.inv_link.fun(lin_param)

        if return_ui:
            if alpha < 0 or alpha > 0.5:
                raise ValueError("`alpha` has to be between 0 and 0.5")
            vcov = self._model.opt_vcov
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
        self._model.data.detach_df()
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
