import numpy as np
from scipy.sparse import coo_matrix

from spxmod.typing import DataFrame, NDArray


class Dimension:
    """Dimension used for grouped variable smoothing.

    Parameters
    ----------
    name : str
        Name of the dimension in the data.

    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._span: NDArray | None = None

    @property
    def span(self) -> NDArray:
        if self._span is None:
            raise ValueError("Dimension values are not set.")
        return self._span

    @property
    def size(self) -> int:
        return len(self.span)

    def set_span(self, data: DataFrame) -> None:
        """Set the unique dimension values.

        Parameters
        ----------
        data : DataFrame
            Data to set the unique dimension values from.

        """
        self._span = np.unique(data[self.name])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name})"


class CategoricalDimension(Dimension):
    """Categorical dimension, mainly used for hierarchical relationships. All
    the values in the span do not have order. The common usage of this dimension
    is to emulate random effect in the linear mixed effect model.

    """


class NumericalDimension(Dimension):
    """Numerical dimension, the values in the space determine the ordered position
    of variable defined on this dimension. This class provide smoothing prior
    constructor used for introduce correlation between variables defined on this
    dimension.

    """

    def build_smoothing_mat(self) -> coo_matrix:
        """Create the smoothing prior matrix for the numerical dimension.

        Returns
        -------
        coo_matrix
            Smoothing prior matrix for the numerical dimension.

        """
        val = np.hstack([np.ones(self.size - 1), -np.ones(self.size - 1)])
        row = np.tile(np.arange(self.size - 1), 2)
        col = np.hstack([np.arange(self.size - 1), np.arange(1, self.size)])
        return coo_matrix((val, (row, col)), shape=(self.size - 1, self.size))

    def build_smoothing_sd(
        self, lam: float, scale_by_distance: bool
    ) -> NDArray:
        """Create the smoothing prior standard deviation vector for the numerical dimension.

        Parameters
        ----------
        lam : float
            Smoothing parameter.
        scale_by_distance : bool
            Whether to scale the prior vector by the distance between the dimension
            values.

        Returns
        -------
        NDArray
            Smoothing prior standard deviation vector for the numerical dimension.

        """
        sd = np.repeat(1 / np.sqrt(lam), self.size - 1)
        if scale_by_distance:
            delta = np.diff(self.span).astype(sd.dtype)
            delta /= delta.min()
            sd *= delta
        return sd


def build_dimension(name: str, dim_type: str) -> Dimension:
    """Create a dimension based on the dimension type.

    Parameters
    ----------
    name : str
        Name of the dimension in the data.
    dim_type : {'categorical', 'numerical'}
        Type of the dimension.

    Returns
    -------
    Dimension
        Dimension based on the dimension type.

    """
    if dim_type == "categorical":
        return CategoricalDimension(name)
    elif dim_type == "numerical":
        return NumericalDimension(name)
    else:
        raise TypeError(f"Dimension type {dim_type} is not supported.")
