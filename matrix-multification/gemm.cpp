// g++ gemm.cpp -std=c++11 -O2 -I /usr/include/eigen3 -DEIGEN_STACK_ALLOCATION_LIMIT=0 -march=native -o gemm  && ./gemm
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <chrono>
#include <ctime>
#include <iostream>

#include <Eigen/Dense>

using Eigen::MatrixXf;

//#define DEBUG

#ifdef DEBUG
  #define N 4
#else
  #define N 512
#endif

float A[N*N];
float B[N*N];
float C[N*N];
float val[N*N];

uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

void print(float arr[]) {
  for (size_t i=0; i < N; i++) {
    for (size_t j=0; j < N; j++) {
      printf("%f ", arr[i*N+j]);
    }
    printf("\n");
  }
}

int load() {
  FILE *f = fopen("/tmp/matmul", "rb");
  if (f == NULL) {
    return 1;
  }
  int BUFFERSIZE = sizeof(float)*N*N;
  fread(A, 1, BUFFERSIZE, f);
  fread(B, 1, BUFFERSIZE, f);
  fread(val, 1, BUFFERSIZE, f);
  fclose(f);
}

void naive_matmul(float A[], float B[], float C[]) {
  for (size_t m=0; m<N; m++) {
    for (size_t n=0; n<N; n++) {
      float acc = 0.0f;
      for (size_t k=0; k<N; k++) {
        acc += A[m*N+k] * B[k*N+n];
      }
      C[m*N+n] = acc;
    }
  }
}

#define BLOCK 4
void tiling_matmul(float A[], float B[], float C[]) {
  for (size_t m=0; m<N; m++) {
    for (size_t n=0; n<N; n++) {
      float acc = 0.0f;
      for (size_t k=0; k<N; k++) {
        acc += A[m*N+k] * B[k*N+n];
      }
      C[m*N+n] = acc;
    }
  }
}

int main()
{
  // load from file
  if (load()) {
    printf("please pregenerate python /tmp/matmul file\n");
    return 1;
  }
  uint64_t start, end;
  double s;
  double gflop = (2.0*N*N*N)*1e-9;
  // naive matmul
  for (int i=0; i<5; i++) {
    start = nanos();
    naive_matmul(A, B, C);
    end = nanos();
    s = (end-start)*1e-9;
    printf("%f GFLOP/S -- %.2f ms\n", gflop/s, s*1e3);
  }
  printf("\n");

  // eigen matmul
  Eigen::Map<Eigen::Matrix<float, N, N, Eigen::RowMajor> > Ae(A);
  Eigen::Map<Eigen::Matrix<float, N, N, Eigen::RowMajor> > Be(B);
  Eigen::Matrix<float, N, N, Eigen::RowMajor> Ce;
  for (int i=0; i<5; i++) {
    start = nanos();
    Ce = Ae * Be;
    end = nanos();
    s = (end-start)*1e-9;
    printf("%f GFLOP/S -- %.2f ms\n", gflop/s, s*1e3);
  }


  // check result
#ifdef DEBUG
  print(C);
#endif
  for (size_t i=0; i<N*N; i++) {
    float v = C[i];
    if (fabsf(v - val[i]) > 1e-3) {
      printf("MISMATCH AT %d, %f != %f\n", i, v, val[i]);
      return 1;
    }
  }
  printf("naive match\n");
  for (size_t i=0; i<N*N; i++) {
    float v = Ce.coeff(i);
    if (fabsf(v - val[i]) > 1e-3) {
      printf("MISMATCH AT %d, %f != %f\n", i, v, val[i]);
      return 1;
    }
  }
  printf("eigen match\n");


  // Eigen test
  std::vector<float> xx = {1.0f, 2.0f, 3.0f, 4.0f};
  auto XXX = Eigen::Map<Eigen::Matrix<float, 2, 2, Eigen::RowMajor> >(xx.data());
  std::cout << XXX << std::endl;

  printf("-------------\n");
  start = nanos();
  Eigen::MatrixXf W1, B1, W2, B2, W3, B3, W4, B4;
  Eigen::MatrixXf O;
  end = nanos();
  printf("declare: %.8f ms\n", (end-start)*1e-6);

  start = nanos();
  O = Eigen::MatrixXf::Random(1, 20);
  W1 = Eigen::MatrixXf::Random(20, 128);
  B1 = Eigen::MatrixXf::Random(1, 128);
  W2 = Eigen::MatrixXf::Random(128, 64);
  B2 = Eigen::MatrixXf::Random(1, 64);
  W3 = Eigen::MatrixXf::Random(64, 32);
  B3 = Eigen::MatrixXf::Random(1, 32);
  W4 = Eigen::MatrixXf::Random(32, 3);
  B4 = Eigen::MatrixXf::Random(1, 3);
  end = nanos();
  printf("random init: %.8f ms\n", (end-start)*1e-6);

  start = nanos();
  O = (O * W1 + B1).cwiseMax(0.0f);
  O = (O * W2 + B2).cwiseMax(0.0f);
  O = (O * W3 + B3).cwiseMax(0.0f);  // (1, 32)
  O = O * W4 + B4;  // (1, 3)
  //std::cout << "O final: " << O << std::endl;
  end = nanos();
  printf("layer forward: %.8f ms\n", (end-start)*1e-6);

  // ziln
  start = nanos();
  float prob = 1.0f / (1.0f + std::exp(-O.coeff(0)));
  //std::cout << "prob: " << prob << std::endl;
  float loc = O.coeff(1);
  //std::cout << "loc: " << loc << std::endl;
  float scale = std::max(0.0f, std::min(O.coeff(2), 5.0f));
  //std::cout << "scale: " << scale << std::endl;
  scale = std::log(std::exp(scale) + 1.0f);
  //std::cout << "scale softplus: " << scale << std::endl;
  float pred = std::exp(loc + 0.5f * scale * scale);
  //std::cout << "pred: " << pred << std::endl;
  //end = nanos();
  //printf("ziln: %.8f ms\n", (end-start)*1e-6);
  end = nanos();
  printf("ziln %.8f ms\n", (end-start)*1e-6);

  printf("-------------\n");
  return 0;
}
