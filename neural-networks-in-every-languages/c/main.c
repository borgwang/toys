// gcc main.c -o main.o -O2 && ./main.o
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N_EXAMPLES 150
#define N_EPOCH 1000
#define IN_DIM 4
#define OUT_DIM 3
#define HIDDEN_DIM 20
#define BS 30
#define LR 0.0003

typedef struct params {
  float w1[IN_DIM][HIDDEN_DIM];
  float b1[HIDDEN_DIM];
  float w2[HIDDEN_DIM][OUT_DIM];
  float b2[OUT_DIM];
} params_t;

void parse_example(char *line, float *feat, float *label) {
  const char *tok;
  int i = 0;
  for (tok = strtok(line, ","); tok; tok=strtok(NULL, ","), i++) {
    if (i < IN_DIM) {
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

float random_uniform(float left, float right) {
  float a = rand() / (float)RAND_MAX;
  return (a + left) * (right - left);
}

int random_integer(int left, int right) {
  return left + rand() % (right - left);
}

void initialize_params(params_t *params) {
  float bound1 = sqrt(6.0 / (IN_DIM + HIDDEN_DIM));
  for (int col = 0; col < HIDDEN_DIM; col++) {
    params->b1[col] = 0.0;
    for (int row = 0; row < IN_DIM; row++) {
      params->w1[row][col] = random_uniform(-bound1, bound1);
    }
  }
  float bound2 = sqrt(6.0 / (HIDDEN_DIM + OUT_DIM));
  for (int col = 0; col < OUT_DIM; col++) {
    params->b2[col] = 0.0;
    for (int row = 0; row < HIDDEN_DIM; row++) {
      params->w2[row][col] = random_uniform(-bound2, bound2);
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

void matmul(int M, int N, int K, float *A, float *B, float *ret) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      *(ret + m * N + n) = 0.0;
      for (int k = 0; k < K; k++) {
        *(ret + m * N + n) += *(A + m * K + k) * *(B + k * N + n);
      }
    }
  }
}

float op_relu(float x) { return x > 0.0 ? x : 0.0; }
float op_drelu(float x) { return x > 0.0 ? 1.0 : 0.0; }

void unary_elemwise(int M, int N, float *A, float (*op)(float)) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      *(A + i * N + j) = op(*(A + i * N + j));
    }
  }
}

void binary_elemwise(int M, int N, float *A, float *B, float (*op)(float, float)) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      *(A + i * N + j) = op(*(A + i * N + j), *(B + i * N + j));
    }
  }
}

void transpose(int M, int N, float *A, float *ret) {
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

void train(params_t* params, float feats[][IN_DIM], float labels[][OUT_DIM], int start) {
  /* forward */
  float a1[BS][HIDDEN_DIM];
  matmul(BS, HIDDEN_DIM, IN_DIM, &feats[start][0], &(params->w1)[0][0], &a1[0][0]);
  for (int i = 0; i < BS; i++) {
    for (int j = 0; j < HIDDEN_DIM; j++) {
      a1[i][j] += params->b1[j];
    }
  }
  unary_elemwise(BS, HIDDEN_DIM, &a1[0][0], op_relu);
  float a2[BS][OUT_DIM];
  matmul(BS, OUT_DIM, HIDDEN_DIM, &a1[0][0], &(params->w2)[0][0], &a2[0][0]);
  for (int i = 0; i < BS; i++) {
    for (int j = 0; j < OUT_DIM; j++) {
      a2[i][j] += params->b2[j];
    }
  }
  softmax(BS, OUT_DIM, &a2[0][0]);
  /* NLL loss */
  float loss = 0.0;
  for (int i = 0; i < BS; i++) {
    float nll = 0.0;
    for (int j = 0; j < OUT_DIM; j++) {
      nll += a2[i][j] * labels[start + i][j];
    }
    loss += -log(nll);
  }
  loss /= BS;

  /* backward */
  float d_logits[BS][OUT_DIM];
  for (int i = 0; i < BS; i++) {
    for (int j = 0; j < OUT_DIM; j++) {
      d_logits[i][j] = a2[i][j] - labels[start + i][j];
    }
  }
  float a1_t[HIDDEN_DIM][BS];
  transpose(BS, HIDDEN_DIM, &a1[0][0], &a1_t[0][0]);
  float d_w2[HIDDEN_DIM][OUT_DIM];
  matmul(HIDDEN_DIM, OUT_DIM, BS, &a1_t[0][0], &d_logits[0][0], &d_w2[0][0]);
  float d_b2[OUT_DIM] = {0.0};
  for (int i = 0; i < BS; i++) {
    for (int j = 0; j < OUT_DIM; j++) {
      d_b2[j] += d_logits[i][j];
    }
  }

  float d_h1[BS][HIDDEN_DIM];
  matmul(BS, HIDDEN_DIM, OUT_DIM, &d_logits[0][0], &(params->w2)[0][0], &d_h1[0][0]);
  unary_elemwise(BS, HIDDEN_DIM, &d_h1[0][0], op_drelu);

  float x_t[IN_DIM][BS];
  transpose(BS, IN_DIM, &feats[start][0], &x_t[0][0]);
  float d_w1[IN_DIM][HIDDEN_DIM];
  matmul(IN_DIM, HIDDEN_DIM, BS, &x_t[0][0], &d_h1[0][0], &d_w1[0][0]);
  float d_b1[HIDDEN_DIM] = {0.0};
  for (int i = 0; i < IN_DIM; i++) {
    for (int j = 0; j < HIDDEN_DIM; j++) {
      d_b1[j] += d_h1[i][j];
    }
  }

  /* gradient descent */
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

void evaluate(params_t* params, float feats[][IN_DIM], float labels[][OUT_DIM], int start, int test_size) {
  /* forward */
  float a1[test_size][HIDDEN_DIM];
  matmul(BS, HIDDEN_DIM, IN_DIM, &feats[start][0], &(params->w1)[0][0], &a1[0][0]);
  for (int i = 0; i < test_size; i++) {
    for (int j = 0; j < HIDDEN_DIM; j++) {
      a1[i][j] += params->b1[j];
    }
  }
  unary_elemwise(test_size, HIDDEN_DIM, &a1[0][0], op_relu);
  float a2[test_size][OUT_DIM];
  matmul(test_size, OUT_DIM, HIDDEN_DIM, &a1[0][0], &(params->w2)[0][0], &a2[0][0]);
  for (int i = 0; i < test_size; i++) {
    for (int j = 0; j < OUT_DIM; j++) {
      a2[i][j] += params->b2[j];
    }
  }
  softmax(test_size, OUT_DIM, &a2[0][0]);

  /* precision */
  int count = 0;
  for (int i = 0; i < test_size; i++) {
    int pred_label, true_label;
    float max_prob = -INFINITY;
    for (int j = 0; j < OUT_DIM; j++) {
      if (a2[i][j] > max_prob) {
        max_prob = a2[i][j];
        pred_label = j;
      }
      if (labels[start + i][j] == 1.0) {
        true_label = j;
      }
    }
    if (pred_label == true_label) {
      count += 1;
    }
  }
  float precision = count / (float)test_size;
  printf("precision: %f\n", precision);
}

void load_dataset(float feats[][IN_DIM], float labels[][OUT_DIM]) {
  FILE *fp = fopen("../data/iris.data", "r");
  char line[64];
  int i = 0;
  while (fgets(line, 64, fp)) {
    line[strcspn(line, "\n")] = '\0';    // get rid of the trailing \n
    if (strlen(line)) {
      char *dup = strdup(line);
      parse_example(dup, feats[i], labels[i]);
      free(dup);
      i++;
    }
  }
}

void preprocess_dataset(float feats[][IN_DIM], float labels[][OUT_DIM], int train_size) {
  /* shuffle dataset */
  srand(time(NULL));
  for (int i = N_EXAMPLES - 1; i > 0; i--) {
    int a = random_integer(0, i);
    swap(feats[i], feats[a], IN_DIM);
    swap(labels[i], labels[a], OUT_DIM);
  }

  /* normalization */
  for (int j = 0; j < IN_DIM; j++) {
    float mean = 0.0, std = 0.0;
    for (int i = 0; i < train_size; i++) {
      mean += feats[i][j];
    }
    mean /= train_size;
    for (int i = 0; i < train_size; i++) {
      std += pow(feats[i][j] - mean, 2);
    }
    std = sqrt(std / train_size);
    printf("dim %d mean: %f std: %f\n", j, mean, std);

    for (int i = 0; i < N_EXAMPLES; i++) {
      feats[i][j] = (feats[i][j] - mean) / std;
    }
  }
}

int main(void) {
  /* load data */
  int train_size = (int)(N_EXAMPLES * 0.8);
  float feats[N_EXAMPLES][IN_DIM];
  float labels[N_EXAMPLES][OUT_DIM] = {0.0};

  load_dataset(feats, labels);
  preprocess_dataset(feats, labels, train_size);

  params_t params;
  initialize_params(&params);
  for (int epoch = 0; epoch < N_EPOCH; epoch++) {
    for (int step = 0; step < train_size / BS; step++) {
      train(&params, feats, labels, step * BS);
    }
    if (epoch % 100 == 0) {
      printf("epoch %d ", epoch);
      evaluate(&params, feats, labels, train_size, N_EXAMPLES - train_size);
    }
  }
  return 0;
}
