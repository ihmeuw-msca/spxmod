# spxmod

Coefficient-smoothing regression models.

`spxmod` fits a [RegMod](https://github.com/ihmeuw-msca/regmod) generalized
linear model (binomial, Poisson or Gaussian) in which each covariate can have a
**different coefficient for every cell of a grid**, with priors that keep
neighboring coefficients close. The grid, called a *space*, is built from one
or more *dimensions* found in the data, such as age, year or location.
Numerical dimensions get a smoothing prior on differences between neighboring
cells; categorical dimensions get a shrinkage prior toward a shared mean. The
design matrix is assembled as a sparse matrix and solved with the Newton and
interior-point solvers from [msca](https://github.com/ihmeuw-msca/msca).

## Install

```bash
pip install git+https://github.com/ihmeuw-msca/spxmod.git
```

Requires Python 3.11 or 3.12. On macOS, `msca` has no PyPI wheel and must be
installed from source first:

```bash
pip install git+https://github.com/ihmeuw-msca/msca.git
pip install git+https://github.com/ihmeuw-msca/spxmod.git
```

## Example

Fit a binomial model where the intercept varies smoothly by age, the effect of
`x` varies by location, and a second `x` effect is shared across all rows.

```python
import numpy as np
import pandas as pd
from spxmod import XModel

rng = np.random.default_rng(0)
df = pd.DataFrame(
    {
        "age": rng.choice([1.0, 2.0, 3.0, 4.0], 200),
        "loc": rng.choice([1, 2, 3], 200),
        "x": rng.normal(size=200),
        "weight": 20.0,
    }
)
p = 1 / (1 + np.exp(-(0.2 * df["age"] + 0.5 * df["x"])))
df["obs"] = rng.binomial(20, p) / 20

model = XModel.from_config(
    {
        "model_type": "binomial",
        "obs": "obs",
        "weights": "weight",
        "spaces": [
            {"dims": [{"name": "age", "dim_type": "numerical"}]},
            {"dims": [{"name": "loc", "dim_type": "categorical"}]},
        ],
        "var_builders": [
            {"name": "intercept", "space": "age", "lam": 1.0, "lam_mean": 1.0},
            {"name": "x", "space": "loc", "lam": 1.0},
            {"name": "x"},
        ],
    }
)
model.fit(df)
pred = model.predict(df)
pred, lower, upper = model.predict(df, return_ui=True, alpha=0.05)
```

## Configuration

`XModel.from_config` takes a dictionary with the following keys.

| Key | Type | Description |
| --- | --- | --- |
| `model_type` | `"binomial"`, `"poisson"`, `"gaussian"` | Likelihood family. |
| `obs` | str | Observation column. For binomial, a rate in `[0, 1]`. |
| `weights` | str | Weight column. For binomial, the sample size. Default `"weight"`. |
| `spaces` | list of space configs | Grids that variables can be partitioned over. |
| `var_builders` | list of variable configs | The variables to fit. |
| `param_specs` | dict, optional | Extra arguments forwarded to the RegMod parameter, e.g. `{"offset": "log_pop"}`. |

### Space

| Key | Type | Description |
| --- | --- | --- |
| `name` | str, optional | Defaults to the dimension names joined with `*`, e.g. `"age*loc"`. |
| `dims` | list of dimension configs | One entry per dimension. |

### Dimension

| Key | Type | Description |
| --- | --- | --- |
| `name` | str | Column in the data. |
| `dim_type` | `"numerical"` or `"categorical"` | Numerical dimensions are ordered and can be smoothed; categorical ones are not. |
| `interval` | `(lb_col, ub_col)`, optional | For numerical dimensions whose rows cover a range (e.g. age groups). Rows are spread over the cells they overlap. |
| `skipna` | bool | Drop rows with missing values when building the grid. Default `True`. |

### Variable

| Key | Type | Description |
| --- | --- | --- |
| `name` | str | Covariate column, or `"intercept"`. |
| `space` | str, optional | Name of a space. If omitted, a single coefficient is fit. |
| `lam` | float or `{dim: float}` | Smoothing strength. Numerical dimension: Gaussian prior with sd `1/sqrt(lam)` on neighboring differences. Categorical: Gaussian prior with sd `1/sqrt(lam)` on each coefficient. Default `0`. |
| `lam_mean` | float | Gaussian prior with sd `1/sqrt(lam_mean)` on the mean of the coefficients. Default `0`. |
| `scale_by_distance` | bool | Scale the smoothing sd by the gap between neighboring values. Default `False`. |
| `gprior` | `{"mean": m, "sd": s}` | Gaussian prior on each coefficient. Scalars or per-cell lists. |
| `uprior` | `{"lb": l, "ub": u}` | Bounds on each coefficient. Scalars or per-cell lists. |
| `order_dim`, `order` | str, list of index lists | Constrain coefficients to be non-decreasing along `order` in dimension `order_dim`. |
| `spline` | dict, optional | `xspline.XSpline` arguments (`knots`, `degree`, ...). The covariate enters through a spline basis instead of linearly. |
| `spline_gpriors`, `spline_upriors` | list of dict, optional | RegMod spline priors. |

### Fitting and prediction

```python
model.fit(df, data_span=None, density=None, **optimizer_options)
model.predict(df, density=None, return_ui=False, alpha=0.05)
```

- `data_span`: a frame used to build the grid instead of `df`, for cells not
  present in the training data.
- `density`: `{(variable_name, space_name): pandas.Series}` indexed by the
  space dimensions. Reweights how an interval row is spread across cells.
- `optimizer_options`: forwarded to the msca solver, e.g. `maxiter`, `gtol`,
  `verbose`. Pass `direct=True` to use a direct Newton solve instead of
  conjugate gradient.
- `return_ui=True` returns a `(3, n)` array of point prediction, lower and
  upper bounds at level `alpha`.
