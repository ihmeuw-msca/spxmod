import numpy as np
cimport numpy as cnp
cimport cython
cimport scipy.linalg.cython_blas as blas

@cython.boundscheck(False)
@cython.wraparound(False)
def get_pred_var(double[:, ::1] mat_view, double[:, ::1] vcov_view):
    cdef int nrow = mat_view.shape[0]
    cdef int ncol = mat_view.shape[1]
    cdef int one_int = 1
    cdef double zero_double = 0
    cdef double one_double = 1

    vec = np.empty(nrow, dtype=np.float_)
    tmp = np.empty(ncol, dtype=np.float_)

    cdef int i
    cdef double[::1] vec_view = vec
    cdef double[::1] tmp_view = tmp

    for i in range(nrow):
        blas.dgemv(
            "N",
            &ncol, &ncol, &one_double,
            &vcov_view[0][0], &ncol,
            &mat_view[i][0], &one_int, &zero_double,
            &tmp_view[0], &one_int
        )

        vec_view[i] = blas.ddot(
            &ncol,
            &tmp_view[0], &one_int,
            &mat_view[i][0], &one_int,
        )

    return vec