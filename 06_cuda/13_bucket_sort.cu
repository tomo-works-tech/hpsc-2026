#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void init_bucket(int *bucket, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < range) {
        bucket[i] = 0;
    }
}

__global__ void count_bucket(int *key, int *bucket, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(&bucket[key[i]], 1);
    }
}

__global__ void write_back(int *key, int *bucket, int range) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= range) return;

    int start = 0;
    for (int k = 0; k < i; k++) {
        start += bucket[k];
    }

    for (int j = 0; j < bucket[i]; j++) {
        key[start + j] = i;
    }
}

int main() {
  int n = 50;
  int range = 5;
  int *key;
  int *bucket;
  int threads = 64;
  cudaMallocManaged(&key, n * sizeof(int));
  cudaMallocManaged(&bucket, range * sizeof(int));

  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  init_bucket<<<(range + threads - 1) / threads, threads>>>(bucket, range);
  cudaDeviceSynchronize();


  count_bucket<<<(n + threads - 1) / threads, threads>>>(key, bucket, n);
  cudaDeviceSynchronize();

  write_back<<<(range + threads - 1) / threads, threads>>>(key, bucket, range);
  cudaDeviceSynchronize();
  
  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
  cudaFree(bucket);
}
