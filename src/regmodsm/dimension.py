import itertools
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.sparse import csc_matrix


class Dimension:
    """Dimension used for grouped variable smoothing.

    Parameters
    ----------
    name : str or list of str
        Name of the dimension column in the data. When it is a list of str, it
        is used for multi-indexing.
    type : str or list of str
        Dimension type. When it is a list of str, it is used for multi-indexing.

    """

    def __init__(self, name: str | list[str], type: str | list[str]) -> None:
        self.name = name
        self.type = type

        self._vals = None
        self._val_index = None
        self.label = "*".join(name) if self.multi_indexed else name

    @property
    def vals(self) -> list[int | float]:
        if self._vals is None:
            raise ValueError("Dimension values are not set.")
        return self._vals

    @property
    def size(self) -> int:
        if self._vals is None:
            raise ValueError("Dimension values are not set.")
        return len(self._vals)

    @property
    def multi_indexed(self) -> bool:
        return not isinstance(self.name, str)

    def set_vals(self, data: DataFrame) -> None:
        """Set the unique dimension values.

        Parameters
        ----------
        data : DataFrame
            Data to set the unique dimension values from.

        """
        self._vals = (
            list(itertools.product(*[np.unique(data[name]) for name in self.name]))
            if self.multi_indexed
            else list(np.unique(data[self.name]))
        )

        self._val_index = {val: i for i, val in enumerate(self._vals)}

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
        index = (
            list(zip(*[data[name] for name in self.name]))
            if self.multi_indexed
            else data[self.name].to_numpy()
        )

        row_index = np.arange(len(data))
        col_index = np.array([self._val_index[val] for val in index])

        mat = csc_matrix((value, (row_index, col_index)), shape=(len(data), self.size))
        columns = [f"{column}_{self.label}_{i}" for i in range(self.size)]
        return pd.DataFrame.sparse.from_spmatrix(mat, index=data.index, columns=columns)
