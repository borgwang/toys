// gcc main.c -o main.o -O2 && ./main.o
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_EXAMPLES 150
#define N_FEATURES 4
#define N_EPOCH 1000
#define IN_DIM 4
#define OUT_DIM 3
#define HIDDEN_DIM 20
#define BS 30
#define LR 0.0003

typedef struct Params {
  float w1[IN_DIM][HIDDEN_DIM];
  float b1[HIDDEN_DIM];
  float w2[HIDDEN_DIM][OUT_DIM];
  float b2[OUT_DIM];
} Params;

void ParseExample(char *line, float *feat, float *label) {
  const char *tok;
  int i = 0;
  for (tok=strtok(line, ","); tok; tok=strtok(NULL, ","), i++) {
    if (i < N_FEATURES) {
      feat[i] = strtof(tok, NULL);
      continue;
    }
    if (strcmp(tok, "Iris-setosa") == 0) {
      label[0] = 1.0;
    } else if (strcmp(tok, "Iris-versicolor") == 0) {
      label[1] = 1.0;
    } else {
      label[2] = 1.0;
    }
  }
}

float GetRandom(float left, float right) {
  float a = rand() / (float)RAND_MAX;
  return (a + left) * (right - left);
}

int GetRandomInteger(int left, int right) {
  return left + rand() % (right - left);
}

void InitializeParams(Params *params) {
  float bound1 = sqrt(6.0 / (IN_DIM + HIDDEN_DIM));
  for (int col = 0; col < HIDDEN_DIM; col++) {
    params->b1[col] = 0.0;
    for (int row = 0; row < IN_DIM; row++) {
      params->w1[row][col] = GetRandom(-bound1, bound1);
    }
  }
  float bound2 = sqrt(6.0 / (HIDDEN_DIM + OUT_DIM));
  for (int col = 0; col < OUT_DIM; col++) {
    params->b2[col] = 0.0;
    for (int row = 0; row < HIDDEN_DIM; row++) {
      params->w2[row][col] = GetRandom(-bound2, bound2);
    }
  }
}

void swap(float *a, float *b, int size) {
  float tmp;
  for (int i = 0; i < size; i++) {
    tmp = *(a + i);
    *(a + i) = *(b + i);
    *(b + i) = tmp;
  }
}

void MatMul(int M, int N, int K, float *A, float *B, float *ret) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      *(ret + m * N + n) = 0.0;
      for (int k = 0; k < K; k++) {
        *(ret + m * N + n) += *(A + m * K + k) * *(B + k * N + n);
      }
    }
  }
}

float OpReLU(float x) { return x > 0.0 ? x : 0.0; }
float OpAdd(float a, float b) { return a + b; }
float OpDReLU(float x) { return x > 0.0 ? 1.0 : 0.0; }

void UnaryElemwise(int M, int N, float *A, float (*op)(float)) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      *(A + i * N + j) = op(*(A + i * N + j));
    }
  }
}

void BinaryElemwise(int M, int N, float *A, float *B, float (*op)(float, float)) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      *(A + i * N + j) = op(*(A + i * N + j), *(B + i * N + j));
    }
  }
}

void MatTranspose(int M, int N, float *A, float *ret) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      *(ret + j * M + i) = *(A + i * N + j);
    }
  }
}

void softmax(int M, int N, float *A) {
  float expCache[N];
  for (int i = 0; i < M; i++) {
    float expSum = 0.0;
    for (int j = 0; j < N; j++) {
      expCache[j] = exp(*(A + i * N + j));
      expSum += expCache[j];
    }
    for (int j = 0; j < OUT_DIM; j++) {
      *(A + i * N + j) = expCache[j] / expSum;
    }
  }
}

void trainBatch(float *X, float *Y, Params *params) {
  // forward
  float a1[BS][HIDDEN_DIM];
  MatMul(BS, HIDDEN_DIM, IN_DIM, X, &(params->w1)[0][0], &a1[0][0]);
  for (int i = 0; i < BS; i++) {
    for (int j = 0; j < HIDDEN_DIM; j++) {
      a1[i][j] += params->b1[j];
    }
  }
  UnaryElemwise(BS, HIDDEN_DIM, &a1[0][0], OpReLU);
  float a2[BS][OUT_DIM];
  MatMul(BS, OUT_DIM, HIDDEN_DIM, &a1[0][0], &(params->w2)[0][0], &a2[0][0]);
  for (int i = 0; i < BS; i++) {
    for (int j = 0; j < OUT_DIM; j++) {
      a2[i][j] += params->b2[j];
    }
  }
  // softmax
  softmax(BS, OUT_DIM, &a2[0][0]);
  // NLL loss
  float loss = 0.0;
  for (int i = 0; i < BS; i++) {
    float nll = 0.0;
    for (int j = 0; j < OUT_DIM; j++) {
      nll += a2[i][j] * (*(Y + i * OUT_DIM + j));
    }
    loss += -log(nll);
  }
  loss /= BS;
  //printf("loss: %f\n", loss);

  // backpropogation
  float d_logits[BS][OUT_DIM];
  for (int i = 0; i < BS; i++) {
    for (int j = 0; j < OUT_DIM; j++) {
      d_logits[i][j] = a2[i][j] - (*(Y + i * OUT_DIM + j));
    }
  }
  float a1_t[HIDDEN_DIM][BS];
  MatTranspose(BS, HIDDEN_DIM, &a1[0][0], &a1_t[0][0]);
  float d_w2[HIDDEN_DIM][OUT_DIM];
  MatMul(HIDDEN_DIM, OUT_DIM, BS, &a1_t[0][0], &d_logits[0][0], &d_w2[0][0]);
  float d_b2[OUT_DIM] = {0.0};
  for (int i = 0; i < BS; i++) {
    for (int j = 0; j < OUT_DIM; j++) {
      d_b2[j] += d_logits[i][j];
    }
  }

  float d_h1[BS][HIDDEN_DIM];
  MatMul(BS, HIDDEN_DIM, OUT_DIM, &d_logits[0][0], &(params->w2)[0][0], &d_h1[0][0]);
  UnaryElemwise(BS, HIDDEN_DIM, &d_h1[0][0], OpDReLU);

  float X_t[IN_DIM][BS];
  MatTranspose(BS, IN_DIM, X, &X_t[0][0]);
  float d_w1[IN_DIM][HIDDEN_DIM];
  MatMul(IN_DIM, HIDDEN_DIM, BS, &X_t[0][0], &d_h1[0][0], &d_w1[0][0]);
  float d_b1[HIDDEN_DIM] = {0.0};
  for (int i = 0; i < IN_DIM; i++) {
    for (int j = 0; j < HIDDEN_DIM; j++) {
      d_b1[j] += d_h1[i][j];
    }
  }

  // gradient descent
  for (int i = 0; i < IN_DIM; i++) {
    for (int j = 0; j < HIDDEN_DIM; j++) {
      params->w1[i][j] -= LR * d_w1[i][j];
    }
  }
  for (int i = 0; i < HIDDEN_DIM; i++) {
    params->b1[i] -= LR * d_b1[i];
  }
  for (int i = 0; i < HIDDEN_DIM; i++) {
    for (int j = 0; j < OUT_DIM; j++) {
      params->w2[i][j] -= LR * d_w2[i][j];
    }
  }
  for (int i = 0; i < OUT_DIM; i++) {
    params->b2[i] -= LR * d_b2[i];
  }
}

void evaluate(float *X, float *Y, Params *params, int testSize) {
  // forward
  float a1[testSize][HIDDEN_DIM];
  MatMul(BS, HIDDEN_DIM, IN_DIM, X, &(params->w1)[0][0], &a1[0][0]);
  for (int i = 0; i < testSize; i++) {
    for (int j = 0; j < HIDDEN_DIM; j++) {
      a1[i][j] += params->b1[j];
    }
  }
  UnaryElemwise(testSize, HIDDEN_DIM, &a1[0][0], OpReLU);
  float a2[testSize][OUT_DIM];
  MatMul(testSize, OUT_DIM, HIDDEN_DIM, &a1[0][0], &(params->w2)[0][0], &a2[0][0]);
  for (int i = 0; i < testSize; i++) {
    for (int j = 0; j < OUT_DIM; j++) {
      a2[i][j] += params->b2[j];
    }
  }
  softmax(testSize, OUT_DIM, &a2[0][0]);
  // precision
  int count = 0;
  for (int i = 0; i < testSize; i++) {
    float maxPred = -INFINITY;
    int predLabel, trueLabel;
    for (int j = 0; j < 3; j++) {
      if (a2[i][j] > maxPred) {
        maxPred = a2[i][j];
        predLabel = j;
      }
      if (*(Y + i * 3 + j) == 1.0) {
        trueLabel = j;
      }
    }
    if (predLabel == trueLabel) {
      count += 1;
    }
  }
  float precision = count / (float)testSize;
  printf("precision: %f\n", precision);
}

int main(void) {
  // load data
  char line[64];
  FILE *fp = fopen("../data/iris.data", "r");
  float feats[N_EXAMPLES][N_FEATURES];
  float labels[N_EXAMPLES][3] = {0.0};
  int i = 0;
  while (fgets(line, 64, fp)) {
    line[strcspn(line, "\n")] = '\0';  // get rid of the trailing \n
    if (strlen(line)) {
      char *dup = strdup(line);
      ParseExample(dup, feats[i], labels[i]);
      free(dup);
      i++;
    }
  }
  // shuffle
  srand(time(NULL));  // random seed
  for (int i = N_EXAMPLES - 1; i > 0; i--) {
    int a = GetRandomInteger(0, i);
    swap(feats[i], feats[a], N_FEATURES);
    swap(labels[i], labels[a], 3);
  }

  // normalization
  int trainSize = (int)(N_EXAMPLES * 0.8);
  for (int j = 0; j < N_FEATURES; j++) {
    // calculate mean and std from the train set
    float mean = 0.0, std = 0.0;
    for (int i = 0; i < trainSize; i++) {
      mean += feats[i][j];
    }
    mean /= trainSize;
    for (int i = 0; i < trainSize; i++) {
      std += pow(feats[i][j] - mean, 2);
    }
    std = sqrt(std / trainSize);
    printf("dim %d mean: %f std: %f\n", i, mean, std);

    for (int i = 0; i < N_EXAMPLES; i++) {
      feats[i][j] = (feats[i][j] - mean) / std;
    }
  }

  Params params;
  InitializeParams(&params);
  for (int epoch = 0; epoch < N_EPOCH; epoch++) {
    for (int step = 0; step < trainSize / BS; step++) {
      trainBatch(&feats[step * BS][0], &labels[step * BS][0], &params);
    }
    if (epoch % 100 == 0) {
      printf("epoch %d ", epoch);
      evaluate(&feats[trainSize][0], &labels[trainSize][0], &params, N_EXAMPLES - trainSize);
    }
  }
  return 0;
}
