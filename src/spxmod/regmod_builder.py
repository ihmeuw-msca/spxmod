from functools import cached_property

import numpy as np
import scipy.sparse as sp
from msca.linalg.matrix import Matrix, asmatrix
from msca.optim.solver import IPSolver, NTCGSolver
from regmod.data import Data
from regmod.models import BinomialModel, GaussianModel, PoissonModel
from regmod.parameter import Parameter
from regmod.prior import (
    GaussianPrior,
    LinearGaussianPrior,
    LinearUniformPrior,
    SplineGaussianPrior,
    SplinePrior,
    SplineUniformPrior,
    UniformPrior,
)
from regmod.variable import SplineVariable, Variable
from scipy.stats import norm
from xspline import XSpline

from spxmod.linalg import get_pred_var
from spxmod.typing import Callable, DataFrame, NDArray, RegmodModel


def msca_optimize(
    model: RegmodModel, x0: NDArray | None = None, options: dict | None = None
) -> NDArray:
    x0 = np.zeros(model.size) if x0 is None else x0
    options = options or {}

    if model.cmat.size == 0:
        solver = NTCGSolver(model.objective, model.gradient, model.hessian)
    else:
        solver = IPSolver(
            model.objective,
            model.gradient,
            model.hessian,
            model.cmat,
            model.cvec,
        )
    result = solver.minimize(x0=x0, **options)
    model.opt_result = result
    model.opt_coefs = result.x.copy()
    model.opt_hessian = model.hessian(model.opt_coefs)
    model.opt_jacobian2 = model.jacobian2(model.opt_coefs)
    return result.x


class SparseParameter(Parameter):
    def get_linear_gmat(self) -> Matrix:
        """Get the linear Gaussian prior matrix.

        Returns
        -------
        Matrix
            Gaussian prior design matrix.

        """
        if len(self.variables) == 0:
            return np.empty(shape=(0, 0))
        gmat = sp.block_diag(
            [
                (
                    var.get_linear_gmat()
                    if isinstance(var, SplineVariable)
                    else np.empty((0, 1))
                )
                for var in self.variables
            ]
        )
        if len(self.linear_gpriors) > 0:
            gmat = sp.vstack(
                [gmat] + [prior.mat for prior in self.linear_gpriors]
            )
        return asmatrix(sp.csc_matrix(gmat))

    def get_linear_umat(self) -> Matrix:
        """Get the linear Uniform prior matrix.

        Returns
        -------
        Matrix
            Uniform prior design matrix.

        """
        if len(self.variables) == 0:
            return np.empty(shape=(0, 0))
        umat = sp.block_diag(
            [
                (
                    var.get_linear_umat()
                    if isinstance(var, SplineVariable)
                    else np.empty((0, 1))
                )
                for var in self.variables
            ]
        )
        if len(self.linear_upriors) > 0:
            umat = sp.vstack(
                [umat] + [prior.mat for prior in self.linear_upriors]
            )
        return asmatrix(sp.csc_matrix(umat))


class SparseRegmodModel(RegmodModel):
    def __init__(
        self,
        data: dict,
        variables: list[dict],
        linear_gpriors: list[dict] | None = None,
        linear_upriors: list[dict] | None = None,
        **kwargs,
    ):
        data = Data(**data)

        # build variables
        variables = [_build_regmod_variable(**v) for v in variables]

        linear_gpriors = linear_gpriors or []
        linear_upriors = linear_upriors or []

        # build smoothing prior
        if linear_gpriors:
            for i, prior in enumerate(linear_gpriors):
                if prior["mat"].size > 0:
                    linear_gpriors[i] = LinearGaussianPrior(**prior)

        if linear_upriors:
            for i, prior in enumerate(linear_upriors):
                if prior["mat"].size > 0:
                    linear_upriors[i] = LinearUniformPrior(**prior)

        param = SparseParameter(
            name=self.param_names[0],
            variables=variables,
            linear_gpriors=linear_gpriors,
            linear_upriors=linear_upriors,
            **kwargs,
        )
        super().__init__(data, params=[param])

    def attach_df(self, df: DataFrame, encode: Callable) -> None:
        param = self.params[0]
        self.data.attach_df(df)
        self.mat = [asmatrix(sp.csc_matrix(encode(df)))]
        self.uvec = param.get_uvec()
        self.gvec = param.get_gvec()
        self.linear_uvec = param.get_linear_uvec()
        self.linear_gvec = param.get_linear_gvec()
        self.linear_gmat = param.get_linear_gmat()
        self.linear_umat = param.get_linear_umat()

        # parse constraints
        cmat = asmatrix(
            sp.vstack(
                [sp.identity(self.mat[0].shape[1]), self.linear_umat],
                format="csr",
            )
        )
        cvec = np.hstack([self.uvec, self.linear_uvec])

        if cmat.size > 0:
            scale = abs(cmat).max(axis=1).toarray().ravel()
            valid = ~np.isclose(scale, 0.0)
            cmat, cvec, scale = cmat[valid], cvec[:, valid], scale[valid]
            if scale.size > 0:
                cmat = cmat.scale_rows(1.0 / scale)
                cvec = cvec / scale

                neg_valid = ~np.isneginf(cvec[0])
                pos_valid = ~np.isposinf(cvec[1])
                cmat = sp.vstack(
                    [-cmat[neg_valid], cmat[pos_valid]], format="csr"
                )
                cvec = np.hstack([-cvec[0][neg_valid], cvec[1][pos_valid]])

        self.cmat = asmatrix(sp.csc_matrix(cmat))
        self.cvec = cvec

    def detach_df(self) -> None:
        self.data.detach_df()
        del self.mat
        del self.uvec
        del self.gvec
        del self.linear_uvec
        del self.linear_gvec
        del self.linear_umat
        del self.linear_gmat
        del self.cmat
        del self.cvec

    def get_lin_param(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        lin_param = mat.dot(coefs)
        if self.params[0].offset is not None:
            lin_param += self.data.get_cols(self.params[0].offset)
        return lin_param

    @cached_property
    def hessian_from_gprior(self) -> Matrix:
        """Hessian matrix from the Gaussian prior.

        Returns
        -------
        Matrix
            Hessian matrix.

        """
        hess = sp.diags(1.0 / self.gvec[1] ** 2, format="csr")
        if self.linear_gvec.size > 0:
            hess += (
                self.linear_gmat.T.scale_cols(1.0 / self.linear_gvec[1] ** 2)
            ).dot(self.linear_gmat)
        return asmatrix(hess)

    def fit(
        self,
        data: DataFrame,
        encode: Callable,
        optimizer: Callable = msca_optimize,
        **optimizer_options,
    ) -> None:
        self.attach_df(data, encode)
        super().fit(optimizer=optimizer, **optimizer_options)
        self.detach_df()

    def predict(
        self,
        data: DataFrame,
        encode: Callable,
        return_ui: bool = False,
        alpha: float = 0.05,
    ) -> NDArray:
        mat = sp.csc_matrix(encode(data))
        param, coef = self.params[0], self.opt_coefs

        offset = np.zeros(len(data))
        if param.offset is not None:
            offset = data[param.offset].to_numpy()

        lin_param = offset + mat.dot(coef)
        pred = param.inv_link.fun(lin_param)

        if return_ui:
            if alpha < 0 or alpha > 0.5:
                raise ValueError("`alpha` has to be between 0 and 0.5")
            # TODO: explore the sparsity of the variance-covariance matrix
            if self.core.size >= 5000:
                raise ValueError(
                    "the number of variables is too large for calculating the "
                    "prediction interval"
                )
            vcov = get_vcov(self.opt_hessian, self.opt_jacobian2)
            lin_param_sd = np.sqrt(get_pred_var(mat, vcov))
            lin_param_lower = norm.ppf(
                0.5 * alpha, loc=lin_param, scale=lin_param_sd
            )
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
        return pred


class SparseBinomialModel(SparseRegmodModel, BinomialModel):
    """Sparse Binomial model."""

    def __init__(self, *args, **kwargs):
        kwargs["inv_link"] = "expit"
        SparseRegmodModel.__init__(self, *args, **kwargs)

    def objective(self, coefs: NDArray) -> float:
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)

        prior_obj = self.objective_from_gprior(coefs)
        likli_obj = weights.dot(
            np.log(1 + np.exp(-y)) + (1 - self.data.obs) * y
        )
        return prior_obj + likli_obj

    def gradient(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))

        prior_grad = self.gradient_from_gprior(coefs)
        likli_grad = mat.T.dot(weights * (z / (1 + z) - self.data.obs))
        return prior_grad + likli_grad

    def hessian(self, coefs: NDArray) -> Matrix:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))
        likli_hess_scale = weights * (z / ((1 + z) ** 2))

        likli_hess_right = mat.scale_rows(likli_hess_scale)
        likli_hess = mat.T.dot(likli_hess_right)

        return self.hessian_from_gprior + likli_hess

    def jacobian2(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))
        likli_jac_scale = weights * (z / (1 + z) - self.data.obs)

        likli_jac = mat.T.scale_cols(likli_jac_scale)
        likli_jac2 = likli_jac.dot(likli_jac.T)
        return self.hessian_from_gprior + likli_jac2


class SparseGaussianModel(SparseRegmodModel, GaussianModel):
    """Sparse Gaussian model."""

    def __init__(self, *args, **kwargs):
        kwargs["inv_link"] = "identity"
        SparseRegmodModel.__init__(self, *args, **kwargs)

    def objective(self, coefs: NDArray) -> float:
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)

        prior_obj = self.objective_from_gprior(coefs)
        likli_obj = 0.5 * weights.dot((y - self.data.obs) ** 2)
        return prior_obj + likli_obj

    def gradient(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)

        prior_grad = self.gradient_from_gprior(coefs)
        likli_grad = mat.T.dot(weights * (y - self.data.obs))
        return prior_grad + likli_grad

    def hessian(self, coefs: NDArray) -> Matrix:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        likli_hess_scale = weights

        prior_hess = self.hessian_from_gprior
        likli_hess_right = mat.scale_rows(likli_hess_scale)
        likli_hess = mat.T.dot(likli_hess_right)

        return prior_hess + likli_hess

    def jacobian2(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)
        likli_jac_scale = weights * (y - self.data.obs)

        likli_jac = mat.T.scale_cols(likli_jac_scale)
        likli_jac2 = likli_jac.dot(likli_jac.T)
        return self.hessian_from_gprior + likli_jac2


class SparsePoissonModel(SparseRegmodModel, PoissonModel):
    """Sparse Poisson model."""

    def __init__(self, *args, **kwargs):
        kwargs["inv_link"] = "exp"
        SparseRegmodModel.__init__(self, *args, **kwargs)

    def objective(self, coefs: NDArray) -> float:
        weights = self.data.weights * self.data.trim_weights
        y = self.get_lin_param(coefs)
        z = np.exp(y)

        prior_obj = self.objective_from_gprior(coefs)
        likli_obj = weights.dot(z - self.data.obs * y)
        return prior_obj + likli_obj

    def gradient(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))

        prior_grad = self.gradient_from_gprior(coefs)
        likli_grad = mat.T.dot(weights * (z - self.data.obs))
        return prior_grad + likli_grad

    def hessian(self, coefs: NDArray) -> Matrix:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))
        likli_hess_scale = weights * z

        prior_hess = self.hessian_from_gprior
        likli_hess_right = mat.scale_rows(likli_hess_scale)
        likli_hess = mat.T.dot(likli_hess_right)

        return prior_hess + likli_hess

    def jacobian2(self, coefs: NDArray) -> NDArray:
        mat = self.mat[0]
        weights = self.data.weights * self.data.trim_weights
        z = np.exp(self.get_lin_param(coefs))
        likli_jac_scale = weights * (z - self.data.obs)

        likli_jac = mat.T.scale_cols(likli_jac_scale)
        likli_jac2 = likli_jac.dot(likli_jac.T)
        return self.hessian_from_gprior + likli_jac2


_model_dict = {
    "binomial": SparseBinomialModel,
    "gaussian": SparseGaussianModel,
    "poisson": SparsePoissonModel,
}


def build_regmod_model(
    model_type: str,
    data: dict,
    variables: list[dict],
    linear_gpriors: list[dict],
    linear_upriors: list[dict],
    param_specs: dict,
) -> RegmodModel:
    # build data
    data = Data(**data)

    # build variables
    variables = [_build_regmod_variable(**kwargs) for kwargs in variables]

    # build smoothing prior
    linear_gpriors_valid = []
    for i, prior in enumerate(linear_gpriors):
        if prior["mat"].size > 0:
            linear_gpriors_valid.append(LinearGaussianPrior(**prior))

    # build order prior
    linear_upriors_valid = []
    for i, prior in enumerate(linear_upriors):
        if prior["mat"].size > 0:
            linear_upriors_valid.append(LinearUniformPrior(**prior))

    # buid regmod model
    model_class = _model_dict[model_type]
    model_param = model_class.param_names[0]

    return model_class(
        data,
        param_specs={
            model_param: {
                "variables": variables,
                "linear_gpriors": linear_gpriors_valid,
                "linear_upriors": linear_upriors_valid,
                **param_specs,
            }
        },
    )


def _build_regmod_variable(
    name: str,
    gprior: dict | None = None,
    uprior: dict | None = None,
    spline: XSpline | None = None,
    spline_gpriors: list[dict] | None = None,
    spline_upriors: list[dict] | None = None,
) -> Variable:
    priors = []
    if gprior is not None:
        priors.append(GaussianPrior(**gprior))
    if uprior is not None:
        priors.append(UniformPrior(**uprior))
    if spline_gpriors is not None:
        priors += [
            SplineGaussianPrior(**spline_gprior)
            for spline_gprior in spline_gpriors
        ]
    if spline_upriors is not None:
        priors += [
            SplineUniformPrior(**spline_uprior)
            for spline_uprior in spline_upriors
        ]
    if spline is None:
        return Variable(name=name, priors=priors)
    else:
        for prior in priors:
            if isinstance(prior, SplinePrior):
                prior.attach_spline(spline)
        return SplineVariable(name=name, spline=spline, priors=priors)


def get_vcov(hessian: Matrix, jacobian2: Matrix) -> NDArray:
    hessian, jacobian2 = hessian.to_numpy(), jacobian2.to_numpy()

    # inverse hessian
    eig_vals, eig_vecs = np.linalg.eigh(hessian)
    if np.isclose(eig_vals, 0.0).any():
        raise ValueError(
            "singular Hessian matrix, please add priors or "
            "reduce number of variables"
        )
    inv_hessian = (eig_vecs / eig_vals).dot(eig_vecs.T)

    # inspect jacobian2
    eig_vals = np.linalg.eigvalsh(jacobian2)
    if np.isclose(eig_vals, 0.0).any():
        raise ValueError(
            "singular Jacobian matrix, please add priors or "
            "reduce number of variables"
        )

    vcov = inv_hessian.dot(jacobian2)
    vcov = inv_hessian.dot(vcov.T)

    return vcov
