from __future__ import annotations

import numpy as np

from spxmod.dimension import CategoricalDimension
from spxmod.space import Space
from spxmod.typing import DataFrame, NDArray


class VariableBuilder:
    """Variable builder to build encoded variables based on the provided space
    and prior information.

    Parameters
    ----------
    name : str
        Name of the variable column in the data.
    space : Space, optional
        Space to partition the variable on. If None, the variable is
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
        name: str,
        space: Space = Space(),
        lam: float | dict[str, float] = 0.0,
        lam_mean: float = 0.0,
        order_dim: str = "",
        order: list[list[int]] | None = None,
        gprior: dict[str, float] | None = None,
        uprior: dict[str, float] | None = None,
        scale_by_distance: bool = False,
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
            self.gprior["sd"] = 1.0 / np.sqrt(lam_cat)

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
        return self.space.size

    def build_variables(self) -> list[dict]:
        """Returns the list of variables in the variable group."""
        variables = [
            dict(name=name, gprior=self.gprior, uprior=self.uprior)
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
        return self.space.build_smoothing_prior(
            self.lam, self.lam_mean, self.scale_by_distance
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
        return self.space.encode(data, column=self.name)
