#include "gmx_blas.h"

void 
F77_FUNC(dtrmm,DTRMM)(char *side, 
                      char *uplo, 
                      char *transa, 
                      char *diag, 
                      int *m, 
                      int *n, 
                      double *alpha, 
                      double *a, 
                      int *lda, 
                      double *b, 
                      int *ldb)
{
    int a_dim1, a_offset, b_dim1, b_offset, i__1, i__2, i__3;

    /* Local variables */
    int i__, j, k, info;
    double temp;
    int lside;
    int nrowa;
    int upper;
    int nounit;
    a_dim1 = *lda;
    a_offset = 1 + a_dim1;
    a -= a_offset;
    b_dim1 = *ldb;
    b_offset = 1 + b_dim1;
    b -= b_offset;

    /* Function Body */
    lside = (*side=='L' || *side=='l');
    if (lside) {
	nrowa = *m;
    } else {
	nrowa = *n;
    }
    nounit = (*diag=='N' || *diag=='n');
    upper = (*uplo=='U' || *uplo=='u');

    info = 0;

    if (*n == 0) {
	return;
    }
    if (*alpha == 0.) {
	i__1 = *n;
	for (j = 1; j <= i__1; ++j) {
	    i__2 = *m;
	    for (i__ = 1; i__ <= i__2; ++i__) {
		b[i__ + j * b_dim1] = 0.;
	    }
	}
	return;
    }
    if (lside) {
	if (*transa=='N' || *transa=='n') {
	    if (upper) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (k = 1; k <= i__2; ++k) {
			if (b[k + j * b_dim1] != 0.) {
			    temp = *alpha * b[k + j * b_dim1];
			    i__3 = k - 1;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] += temp * a[i__ + k * 
					a_dim1];
			    }
			    if (nounit) {
				temp *= a[k + k * a_dim1];
			    }
			    b[k + j * b_dim1] = temp;
			}
		    }
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    for (k = *m; k >= 1; --k) {
			if (b[k + j * b_dim1] != 0.) {
			    temp = *alpha * b[k + j * b_dim1];
			    b[k + j * b_dim1] = temp;
			    if (nounit) {
				b[k + j * b_dim1] *= a[k + k * a_dim1];
			    }
			    i__2 = *m;
			    for (i__ = k + 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] += temp * a[i__ + k * 
					a_dim1];
			    }
			}
		    }
		}
	    }
	} else {

	    if (upper) {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    for (i__ = *m; i__ >= 1; --i__) {
			temp = b[i__ + j * b_dim1];
			if (nounit) {
			    temp *= a[i__ + i__ * a_dim1];
			}
			i__2 = i__ - 1;
			for (k = 1; k <= i__2; ++k) {
			    temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
			}
			b[i__ + j * b_dim1] = *alpha * temp;
		    }
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			temp = b[i__ + j * b_dim1];
			if (nounit) {
			    temp *= a[i__ + i__ * a_dim1];
			}
			i__3 = *m;
			for (k = i__ + 1; k <= i__3; ++k) {
			    temp += a[k + i__ * a_dim1] * b[k + j * b_dim1];
			}
			b[i__ + j * b_dim1] = *alpha * temp;
		    }
		}
	    }
	}
    } else {
	if (*transa=='N' || *transa=='n') {

	    if (upper) {
		for (j = *n; j >= 1; --j) {
		    temp = *alpha;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__1 = *m;
		    for (i__ = 1; i__ <= i__1; ++i__) {
			b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
		    }
		    i__1 = j - 1;
		    for (k = 1; k <= i__1; ++k) {
			if (a[k + j * a_dim1] != 0.) {
			    temp = *alpha * a[k + j * a_dim1];
			    i__2 = *m;
			    for (i__ = 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k * 
					b_dim1];
			    }
			}
		    }
		}
	    } else {
		i__1 = *n;
		for (j = 1; j <= i__1; ++j) {
		    temp = *alpha;
		    if (nounit) {
			temp *= a[j + j * a_dim1];
		    }
		    i__2 = *m;
		    for (i__ = 1; i__ <= i__2; ++i__) {
			b[i__ + j * b_dim1] = temp * b[i__ + j * b_dim1];
		    }
		    i__2 = *n;
		    for (k = j + 1; k <= i__2; ++k) {
			if (a[k + j * a_dim1] != 0.) {
			    temp = *alpha * a[k + j * a_dim1];
			    i__3 = *m;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k * 
					b_dim1];
			    }
			}
		    }
		}
	    }
	} else {

	    if (upper) {
		i__1 = *n;
		for (k = 1; k <= i__1; ++k) {
		    i__2 = k - 1;
		    for (j = 1; j <= i__2; ++j) {
			if (a[j + k * a_dim1] != 0.) {
			    temp = *alpha * a[j + k * a_dim1];
			    i__3 = *m;
			    for (i__ = 1; i__ <= i__3; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k * 
					b_dim1];
			    }
			}
		    }
		    temp = *alpha;
		    if (nounit) {
			temp *= a[k + k * a_dim1];
		    }
		    if (temp != 1.) {
			i__2 = *m;
			for (i__ = 1; i__ <= i__2; ++i__) {
			    b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
			}
		    }
		}
	    } else {
		for (k = *n; k >= 1; --k) {
		    i__1 = *n;
		    for (j = k + 1; j <= i__1; ++j) {
			if (a[j + k * a_dim1] != 0.) {
			    temp = *alpha * a[j + k * a_dim1];
			    i__2 = *m;
			    for (i__ = 1; i__ <= i__2; ++i__) {
				b[i__ + j * b_dim1] += temp * b[i__ + k * 
					b_dim1];
			    }
			}
		    }
		    temp = *alpha;
		    if (nounit) {
			temp *= a[k + k * a_dim1];
		    }
		    if (temp != 1.) {
			i__1 = *m;
			for (i__ = 1; i__ <= i__1; ++i__) {
			    b[i__ + k * b_dim1] = temp * b[i__ + k * b_dim1];
			}
		    }
		}
	    }
	}
    }

    return;

}


