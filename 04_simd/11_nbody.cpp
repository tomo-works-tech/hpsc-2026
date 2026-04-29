#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>
int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    __m512  xvec = _mm512_load_ps(x);
    __m512  yvec= _mm512_load_ps(y);
    __m512  mvec= _mm512_load_ps(m);
    __m512  xi = _mm512_set1_ps(x[i]);
    __m512  yi = _mm512_set1_ps(y[i]);
    __m512i j_idx = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m512i i_idx = _mm512_set1_epi32(i);
    __mmask16 mask = _mm512_cmp_epi32_mask(i_idx, j_idx, _MM_CMPINT_NE);

    __m512 rx = _mm512_sub_ps(xi, xvec);   
    __m512 ry = _mm512_sub_ps(yi, yvec);        

    __m512 r2 = _mm512_add_ps(_mm512_mul_ps(rx, rx), _mm512_mul_ps(ry, ry)); 
    __m512 inv_r  = _mm512_rsqrt14_ps(r2);  
    __m512 inv_r3 = _mm512_mul_ps(inv_r, _mm512_mul_ps(inv_r, inv_r));

    __m512 fxvec = _mm512_mul_ps(_mm512_mul_ps(rx, mvec), inv_r3);
    __m512 fyvec = _mm512_mul_ps(_mm512_mul_ps(ry, mvec), inv_r3);

    fx[i] -= _mm512_mask_reduce_add_ps(mask, fxvec);
    fy[i] -= _mm512_mask_reduce_add_ps(mask, fyvec);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
