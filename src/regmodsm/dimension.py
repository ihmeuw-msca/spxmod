import itertools
import functools
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.sparse import csc_matrix, identity, kron, vstack


class Dimension:
    """Dimension used for grouped variable smoothing.

    Parameters
    ----------
    name : str or list of str
        Name of the dimension column(s) in the data. When it is a list
        of str, it is used for multi-indexing.
    type : str or list of str
        Dimension type(s), either "numerical" or "categorical". When it
        is a list of str, it is used for multi-indexing.
    label : str, optional
        Name of dimension in the model.

    """

    def __init__(
        self,
        name: str | list[str],
        type: str | list[str],
        label: str | None = None,
    ) -> None:
        self.name = [name] if isinstance(name, str) else list(name)
        self.type = [type] if isinstance(type, str) else list(type)
        self.label = "*".join(self.name) if label is None else str(label)

        self._vals = None
        self._unique_vals = None

    @property
    def vals(self) -> DataFrame:
        if self._vals is None:
            raise ValueError("Dimension values are not set.")
        return self._vals

    @property
    def size(self) -> int:
        if self._vals is None:
            raise ValueError("Dimension values are not set.")
        return len(self._vals)

    def set_vals(self, data: DataFrame) -> None:
        """Set the unique dimension values.

        Parameters
        ----------
        data : DataFrame
            Data to set the unique dimension values from.

        """
        self._unique_vals = {name: np.unique(data[name]) for name in self.name}
        combinations = list(itertools.product(*self._unique_vals.values()))
        self._vals = pd.DataFrame(combinations, columns=self.name).reset_index()

    def get_dummy_names(self, column: str) -> list[str]:
        """Get the dummy variable names for the dimension.

        Parameters
        ----------
        column : str
            Column name to use for the dummy variable names.

        Returns
        -------
        list of str
            Dummy variable names for the dimension.

        """
        return [f"{column}_{self.label}_{i}" for i in range(self.size)]

    def get_dummies(self, data: DataFrame, column: str = "intercept") -> DataFrame:
        """Get the dummy variables for the dimension.

        Parameters
        ----------
        data : DataFrame
            Data to get the dummy variables from.
        column : str, default "intercept"
            Column to use for the dummy variable values. Default is "intercept".

        Returns
        -------
        DataFrame
            Dummy variables data frame for the dimension.

        """
        value = data[column].to_numpy()

        row_index = np.arange(len(data))
        col_index = (
            data[self.name]
            .merge(self.vals, how="left", on=self.name)["index"]
            .to_numpy()
        )

        mat = csc_matrix((value, (row_index, col_index)), shape=(len(data), self.size))
        columns = self.get_dummy_names(column)
        return pd.DataFrame.sparse.from_spmatrix(mat, index=data.index, columns=columns)

    def get_smoothing_gprior(
        self, lam: dict[str, float], scale_by_distance: bool = False
    ) -> tuple[NDArray, NDArray]:
        """Get the smoothing Gaussian prior for the dimension.

        Parameters
        ----------
        lam : dict of float
            Smoothing parameters for the dimension.
        scale_by_distance : bool, default False
            Whether to scale the prior vector by the distance between the dimension
            values. Default is False.

        Returns
        -------
        tuple[NDArray, NDArray]
            Smoothing Gaussian prior matrix and vector.

        """
        mat = csc_matrix((0, self.size), dtype=float)
        vec = np.empty((0,))

        for i, type in enumerate(self.type):
            if type == "numerical":
                sub_mats, sub_vecs = [], []
                for j, name in enumerate(self.name):
                    unique_vals = self._unique_vals[name]
                    size = len(unique_vals)
                    if i == j:
                        sub_mats.append(_get_numerical_gmat(size))
                        sub_vecs.append(
                            _get_numerical_gvec_sd(
                                lam[name], unique_vals, scale_by_distance
                            )
                        )
                    else:
                        sub_mats.append(identity(size))
                        sub_vecs.append(np.ones(size))
                mat = vstack([mat, functools.reduce(kron, sub_mats)])
                vec = np.hstack([vec, functools.reduce(_flatten_outer, sub_vecs)])
        vec = np.vstack([np.zeros(vec.size), vec])
        return mat, vec


def _flatten_outer(x: NDArray, y: NDArray) -> NDArray:
    """Flatten the outer product of two arrays.

    Parameters
    ----------
    x : NDArray
        First array.
    y : NDArray
        Second array.

    Returns
    -------
    NDArray
        Flattened outer product of the two arrays.

    """
    return np.outer(x.ravel(), y.ravel()).ravel()


def _get_numerical_gmat(size: int) -> csc_matrix:
    """Get the numerical Gaussian prior matrix.

    Parameters
    ----------
    size : int
        Size of the Gaussian prior matrix.

    Returns
    -------
    csc_matrix
        Gaussian prior matrix.

    """
    value = np.hstack([np.ones(size - 1), -np.ones(size - 1)])
    row_index = np.tile(np.arange(size - 1), 2)
    col_index = np.hstack([np.arange(size - 1), np.arange(1, size)])
    mat = csc_matrix((value, (row_index, col_index)), shape=(size - 1, size))
    return mat


def _get_numerical_gvec_sd(
    lam: float, vals: NDArray, scale_by_distance: bool
) -> NDArray:
    """Get the numerical Gaussian prior standard deviation.

    Parameters
    ----------
    lam : float
        Smoothing parameter.
    vals : NDArray
        Dimension values.
    scale_by_distance : bool
        Whether to scale the prior vector by the distance between the dimension
        values.

    Returns
    -------
    NDArray
        Gaussian prior standard deviation.

    """
    vec = np.repeat(1 / np.sqrt(lam), len(vals) - 1)
    if scale_by_distance:
        delta = np.diff(vals).astype(vec.dtype)
        delta /= delta.min()
        vec *= delta
    return vec
