#include <ctype.h>
#include "gmx_blas.h"

void
F77_FUNC(ssymv,SSYMV)(char *uplo,
                      int *n,
                      float *alpha,
                      float *a,
                      int *lda,
                      float *x,
                      int *incx,
                      float *beta,
                      float *y,
                      int *incy)
{
  char ch=toupper(*uplo);
  int kx,ky,i,j,ix,iy,jx,jy;
  float temp1,temp2;

  if(*n<=0 || *incx==0 || *incy==0)
    return;
  
  if(*incx>0)
    kx = 1;
  else
    kx = 1 - (*n -1)*(*incx);

  if(*incy>0)
    ky = 1;
  else
    ky = 1 - (*n -1)*(*incy);

  if(*beta != 1.0) {
    if(*incy==1) {
      if(*beta==0.0) 
	for(i=1;i<=*n;i++)
	  y[i-1] = 0.0;
      else
	for(i=1;i<=*n;i++)
	  y[i-1] *= *beta;
    } else {
      /* non-unit incr. */
      iy = ky;
      if(*beta==0.0) 
	for(i=1;i<=*n;i++) {
	  y[iy-1] = 0.0;
	  iy += *incy;
	}
      else
      	for(i=1;i<=*n;i++) {
	  y[iy-1] *= *beta;
	  iy += *incy;
	}
    }
  }
  
  if(*alpha==0.0) 
    return;
  
  if(ch=='U') {
    if(*incx==1 && *incy==1) {
      for(j=1;j<=*n;j++) {
	temp1 = *alpha * x[j-1];
	temp2 = 0.0;
	for(i=1;i<j;i++) {
	  y[i-1] += temp1*a[(j-1)*(*lda)+(i-1)];
	  temp2 += a[(j-1)*(*lda)+(i-1)] * x[i-1];
	}
	y[j-1] += temp1*a[(j-1)*(*lda)+(j-1)] + *alpha *temp2;
      }
    } else {
      /* non-unit incr. */
      jx = kx;
      jy = ky;
      for(j=1;j<=*n;j++) {
	temp1 = *alpha * x[jx-1];
	temp2 = 0.0;
	ix = kx;
	iy = ky;
	for(i=1;i<j;i++) {
	  y[iy-1] += temp1 * a[(j-1)*(*lda)+(i-1)];
	  temp2 += a[(j-1)*(*lda)+(i-1)] * x[ix-1];
	  ix += *incx;
	  iy += *incy;
	}
	y[jy-1] += temp1*a[(j-1)*(*lda)+(j-1)] + *alpha*temp2;
	jx += *incx;
	jy += *incy;
      }
    }
  } else {
    /* lower */
    if(*incx==1 && *incy==1) {
      for(j=1;j<=*n;j++) {
	temp1 = *alpha * x[j-1];
	temp2 = 0.0;
	y[j-1] += temp1 * a[(j-1)*(*lda)+(j-1)];
	for(i=j+1;i<=*n;i++) {
	  y[i-1] += temp1*a[(j-1)*(*lda)+(i-1)];
	  temp2 += a[(j-1)*(*lda)+(i-1)] * x[i-1];
	}
	y[j-1] += *alpha *temp2;
      }
    } else {
      /* non-unit incr. */
      jx = kx;
      jy = ky;
      for(j=1;j<=*n;j++) {
	temp1 = *alpha * x[jx-1];
	temp2 = 0.0;
	y[jy-1] += temp1 * a[(j-1)*(*lda)+(j-1)];
	ix = jx;
	iy = jy;
	for(i=j+1;i<=*n;i++) {
	  ix += *incx;
	  iy += *incy;
	  y[iy-1] += temp1 * a[(j-1)*(*lda)+(i-1)];
	  temp2 += a[(j-1)*(*lda)+(i-1)] * x[ix-1];
	}
	y[jy-1] += *alpha*temp2;
	jx += *incx;
	jy += *incy;
      }
    }
  }
  return;
}    
