import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from regmodsm.dimension import Dimension
from regmod.prior import GaussianPrior, Prior, UniformPrior
from regmod.variable import Variable


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
