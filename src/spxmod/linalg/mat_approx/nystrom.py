import numpy as np
import scipy as sp

from spxmod.typing import NDArray


class NystromApprox:
    """Nystrom approximation for a large PSD matrix/linear operator.
    Provided mat and probe, we try to compute the projected approximation.
    mat<probe> =
      (mat @ probe) @ pseudo_inv(probe.T @ mat @ probe) @ (mat @ probe).T
    For details please check algorithm 2.1 in
        https://arxiv.org/pdf/2110.02820

    Parameters
    ----------
    mat
        Linear operator or matrix to be approximated.
    probe
        Probe matrix to be used for approximation.
    eps
        Shifted parameter for stable pseudo-inverse computation.

    Note
    ----
    If `probe` matrix is not orthonormal, it will be orthonormalized using QR.

    """

    def __init__(
        self,
        mat: sp.sparse.linalg.LinearOperator,
        probe: NDArray,
        eps: float = 1e-16,
    ) -> None:
        if mat.shape[1] != probe.shape[0]:
            raise ValueError("mat and probe must have compatible shapes.")
        self.mat, self.shape = mat, mat.shape
        if not np.allclose(probe.T.dot(probe), np.identity(probe.shape[1])):
            probe = np.linalg.qr(probe, mode="reduced").Q
        self.probe = probe
        self.eps = eps

        self.eigvecs, self.eigvals = self.build_approximation()
        self._diag_eigvals = sp.sparse.diags(self.eigvals)

    def build_approximation(self) -> tuple[NDArray, NDArray]:
        y = self.mat.dot(self.probe)
        nu = self.eps * np.linalg.norm(y, ord="fro")
        y_shifted = y + nu * self.probe
        c = sp.linalg.cholesky(self.probe.T.dot(y_shifted))
        b = sp.linalg.solve_triangular(c.T, y_shifted.T, lower=True).T
        u, s, _ = np.linalg.svd(b, full_matrices=False)
        s = np.maximum(0.0, s**2 - nu)

        return u, s

    def dot(self, x: NDArray) -> NDArray:
        return self.eigvecs @ (self._diag_eigvals @ (self.eigvecs.T @ x))

    def matvec(self, x: NDArray) -> NDArray:
        return self.dot(x)

    def matmat(self, x: NDArray) -> NDArray:
        return self.dot(x)

    def __matmul__(self, x: NDArray) -> NDArray:
        return self.dot(x)
