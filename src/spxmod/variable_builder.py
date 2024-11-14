from __future__ import annotations

import numpy as np
from xspline import XSpline

from spxmod.dimension import CategoricalDimension
from spxmod.space import Space
from spxmod.typing import DataFrame, NDArray


class VariableBuilder:
    """Variable builder to build encoded variables based on the provided space
    and prior information.

    Parameters
    ----------
    name
        Name of the variable column in the data.
    space
        Space to partition the variable on. If None, the variable is
        not partitioned.
    lam
        Regularization parameter for the coefficients in the variable
        group. Default is 0. If the dimension is numerical, a Gaussian
        prior with mean 0 and standard deviation 1/sqrt(lam) is set on
        the differences between neighboring coefficients along the
        dimension. If the dimension is categorical, a Gaussian prior
        with mean 0 and standard deviation 1/sqrt(lam) is set on the
        coefficients.
    lam_mean
        Regularization parameter for the mean of the coefficients in the
        variable group. Default is 1e-8. A Gaussian prior with mean 0
        and standard deviation 1/sqrt(lam_mean) is set on the mean of
        the coefficients.
    gprior
        Gaussian prior for the variable. Default is (0, np.inf).
        Argument is overwritten with (0, 1/sqrt(lam)) if dimension is
        categorical.
    uprior
        Uniform prior for the variable. Default is (-np.inf, np.inf).
    scale_by_distance : bool, optional
        Whether to scale the prior standard deviation by the distance
        between the neighboring values along the dimension. For
        numerical dimensions only. Default is False.
    spline
        Spline configuration for the variable. If None, variable will be parsed
        as an instance of Variable, otherwise, it will be parsed as an instance
        of SplineVariable.
    spline_gpriors
        Gaussian priors for the spline coefficients. For details please check
        https://github.com/ihmeuw-msca/regmod/blob/release/0.1.2/src/regmod/prior.py
    spline_upriors
        Uniform priors for the spline coefficients. For details please check
        https://github.com/ihmeuw-msca/regmod/blob/release/0.1.2/src/regmod/prior.py

    """

    def __init__(
        self,
        name: str,
        space: Space = Space(),
        lam: float | dict[str, float] = 0.0,
        lam_mean: float = 0.0,
        order_dim: str = "",
        order: list[list[int]] | None = None,
        gprior: dict[str, float] | None = None,
        uprior: dict[str, float] | None = None,
        scale_by_distance: bool = False,
        spline: dict | None = None,
        spline_gpriors: list[dict] | None = None,
        spline_upriors: list[dict] | None = None,
    ) -> None:
        self.name = name
        self.space = space
        self.lam = lam
        self.lam_mean = lam_mean
        self.order_dim = order_dim
        self.order = order
        self.gprior = gprior or dict(mean=0.0, sd=np.inf)
        self.uprior = uprior or dict(lb=-np.inf, ub=np.inf)
        self.scale_by_distance = scale_by_distance
        gprior = gprior or dict(mean=0.0, sd=np.inf)
        uprior = uprior or dict(lb=-np.inf, ub=np.inf)
        for prior in [gprior, uprior]:
            for key, value in prior.items():
                if isinstance(value, list):
                    if spline is not None:
                        raise ValueError(
                            "Cannot provide vector prior for spline variable"
                        )
                    prior[key] = np.asarray(value, dtype="float")
        self.gprior = gprior
        self.uprior = uprior

        if spline is not None:
            spline = XSpline(**spline)
        self.spline: XSpline = spline
        self.spline_gpriors = spline_gpriors
        self.spline_upriors = spline_upriors

        # transfer lam to gprior when dim is categorical
        if isinstance(self.lam, float):
            self.lam = {name: self.lam for name in self.space.dim_names}

        # TODO: this behavior is up-to-discussion
        lam_cat = sum(
            [
                self.lam.get(dim.name, 0.0)
                for dim in self.space.dims
                if isinstance(dim, CategoricalDimension)
            ]
        )
        if lam_cat > 0:
            if isinstance(self.gprior["sd"], np.ndarray):
                self.gprior["sd"].fill(1.0 / np.sqrt(lam_cat))
            else:
                self.gprior["sd"] = 1.0 / np.sqrt(lam_cat)
        if self.spline is not None:
            self.gprior["size"] = self.spline.num_spline_bases
            self.uprior["size"] = self.spline.num_spline_bases

    @classmethod
    def from_config(
        cls, config: dict, spaces: dict[str, Space]
    ) -> VariableBuilder:
        space_name = config.get("space")
        if space_name:
            config["space"] = spaces[space_name]
        return cls(**config)

    @property
    def size(self) -> int:
        """Number of variables in the variable group."""
        if self.spline is None:
            return self.space.size
        return self.spline.num_spline_bases * self.space.size

    def build_variables(self) -> list[dict]:
        """Returns the list of variables in the variable group."""
        prior_info = {**self.gprior, **self.uprior}
        for key, value in prior_info.items():
            if np.isscalar(value):
                prior_info[key] = np.repeat(value, self.size)

        if self.spline is None:
            variables = [
                dict(
                    name=name,
                    gprior=dict(
                        mean=prior_info["mean"][i], sd=prior_info["sd"][i]
                    ),
                    uprior=dict(lb=prior_info["lb"][i], ub=prior_info["ub"][i]),
                )
                for i, name in enumerate(
                    self.space.build_encoded_names(self.name)
                )
            ]
        else:
            variables = [
                dict(
                    name=name,
                    gprior=self.gprior,
                    uprior=self.uprior,
                    spline=self.spline,
                    spline_gpriors=self.spline_gpriors,
                    spline_upriors=self.spline_upriors,
                )
                for name in self.space.build_encoded_names(self.name)
            ]
        return variables

    def build_smoothing_prior(self) -> dict[str, NDArray]:
        """Returns the smoothing Gaussian prior for the variable group.

        If the dimension is numerical and lam > 0, a Gaussian prior with
        mean 0 and standard deviation 1/sqrt(lam) is set on the
        differences between neighboring coefficients along the
        dimension. For both dimension types if lam_mean > 0, a Gaussian
        prior with mean 0 and standard deviation 1/sqrt(lam_mean) is set
        on the mean of the coefficients.

        Returns
        -------
        dict[str, NDArray]
            Smoothing Gaussian prior matrix and vector.

        """
        size = 1 if self.spline is None else self.spline.num_spline_bases
        return self.space.build_smoothing_prior(
            size, self.lam, self.lam_mean, self.scale_by_distance
        )

    def build_order_prior(self) -> dict[str, NDArray]:
        return self.space.build_order_prior(self.order_dim, self.order)

    def encode(self, data: DataFrame) -> DataFrame:
        """Encode variable column based on the space.

        Parameters
        ----------
        data : DataFrame
            Data containing the variable column.

        Returns
        -------
        DataFrame
            Encoded variable columns.

        """
        val = (
            np.ones(len(data))
            if self.name == "intercept"
            else data[self.name].to_numpy()
        )

        if self.spline is not None:
            mat = self.spline.design_mat(val)
        else:
            mat = val[:, np.newaxis]

        coords = data[self.space.dim_names]

        return self.space.encode(mat, coords)
