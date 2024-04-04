import itertools
import functools
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.sparse import coo_matrix, csc_matrix, identity, kron, vstack


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

        self._span = None
        self._unique = None
        self._nunique = None

    @property
    def span(self) -> DataFrame:
        if self._span is None:
            raise ValueError("Dimension values are not set.")
        return self._span

    @property
    def size(self) -> int:
        if self._span is None:
            raise ValueError("Dimension values are not set.")
        return len(self._span)

    def set_span(self, data: DataFrame) -> None:
        """Set the unique dimension values.

        Parameters
        ----------
        data : DataFrame
            Data to set the unique dimension values from.

        """
        self._unique = {name: np.unique(data[name]) for name in self.name}
        self._nunique = {name: len(self._unique[name]) for name in self.name}
        combinations = list(itertools.product(*self._unique.values()))
        self._span = pd.DataFrame(combinations, columns=self.name).reset_index()

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
            .merge(self._span, how="left", on=self.name)["index"]
            .to_numpy()
        )

        mat = csc_matrix((value, (row_index, col_index)), shape=(len(data), self.size))
        columns = self.get_dummy_names(column)
        return pd.DataFrame.sparse.from_spmatrix(mat, index=data.index, columns=columns)

    def get_smoothing_gprior(
        self,
        lam: float | dict[str, float],
        lam_mean: float,
        scale_by_distance: bool = False,
    ) -> tuple[NDArray, NDArray]:
        """Get the smoothing Gaussian prior for the dimension.

        Parameters
        ----------
        lam : float or dict of float
            Smoothing parameters for the dimension.
        lam_mean : float
            Smoothing parameter for the mean of the coefficients.
        scale_by_distance : bool, default False
            Whether to scale the prior vector by the distance between the dimension
            values. Default is False.

        Returns
        -------
        tuple[NDArray, NDArray]
            Smoothing Gaussian prior matrix and vector.

        """
        lam = {name: lam for name in self.name} if isinstance(lam, float) else lam
        mat = coo_matrix((0, self.size))
        vec = np.empty((0,))

        sub_mats_default = list(map(identity, self._nunique.values()))
        sub_vecs_default = list(map(np.ones, self._nunique.values()))

        for i, name in enumerate(self.name):
            lam_i = lam.get(name, 0.0)
            if self.type[i] == "numerical" and lam_i > 0.0:
                sub_mats, sub_vecs = sub_mats_default.copy(), sub_vecs_default.copy()
                sub_mats[i] = _get_numerical_gmat(self._nunique[name])
                sub_vecs[i] = _get_numerical_gvec_sd(
                    lam_i, self._unique[name], scale_by_distance
                )
                mat = vstack([mat, functools.reduce(kron, sub_mats)])
                vec = np.hstack([vec, functools.reduce(_flatten_outer, sub_vecs)])

        if lam_mean > 0.0:
            mat = vstack([mat, np.repeat(1 / self.size, self.size)])
            vec = np.hstack([vec, [1 / np.sqrt(lam_mean)]])

        # TODO: regmod cannot recognize sparse array as prior, this shouldn't
        # be necessary in the future
        mat = mat.toarray()
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
    mat = coo_matrix((value, (row_index, col_index)), shape=(size - 1, size))
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
