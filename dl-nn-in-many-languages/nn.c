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
#define LR 0.0003f

#define DATA_PATH "data/iris.data"
#define SPLIT_PATH "data/split.txt"
#define PARAMS_PATH "data/params.txt"

typedef struct params {
  float w1[IN_DIM][HIDDEN_DIM];
  float b1[HIDDEN_DIM];
  float w2[HIDDEN_DIM][OUT_DIM];
  float b2[OUT_DIM];
} params_t;

float random_uniform(float left, float right) {
  float a = rand() / (float)RAND_MAX;
  return (a + left) * (right - left);
}

void initialize_params(params_t *params) {
  float bound1 = sqrtf(6.0f / (IN_DIM + HIDDEN_DIM));
  for (size_t col = 0; col < HIDDEN_DIM; col++) {
    params->b1[col] = 0.0f;
    for (size_t row = 0; row < IN_DIM; row++) {
      params->w1[row][col] = random_uniform(-bound1, bound1);
    }
  }
  float bound2 = sqrtf(6.0f / (HIDDEN_DIM + OUT_DIM));
  for (size_t col = 0; col < OUT_DIM; col++) {
    params->b2[col] = 0.0f;
    for (size_t row = 0; row < HIDDEN_DIM; row++) {
      params->w2[row][col] = random_uniform(-bound2, bound2);
    }
  }
}

void load_params(params_t *params) {
  FILE *fp = fopen(PARAMS_PATH, "r");
  char line[1024];
  char *token;
  if (fgets(line, sizeof(line), fp)) {
    token = strtok(line, ",");
    for (size_t i = 0; i < IN_DIM; i++) {
      for (size_t j = 0; j < HIDDEN_DIM; j++) {
        params->w1[i][j] = atof(token);
        token = strtok(NULL, ",");
      }
    }
  }
  if (fgets(line, sizeof(line), fp)) {
    token = strtok(line, ",");
    for (size_t i = 0; i < HIDDEN_DIM; i++) {
      params->b1[i] = atof(token);
      token = strtok(NULL, ",");
    }
  }
  if (fgets(line, sizeof(line), fp)) {
    token = strtok(line, ",");
    for (size_t i = 0; i < HIDDEN_DIM; i++) {
      for (size_t j = 0; j < OUT_DIM; j++) {
        params->w2[i][j] = atof(token);
        token = strtok(NULL, ",");
      }
    }
  }
  if (fgets(line, sizeof(line), fp)) {
    token = strtok(line, ",");
    for (size_t i = 0; i < OUT_DIM; i++) {
      params->b2[i] = atof(token);
      token = strtok(NULL, ",");
    }
  }
  fclose(fp);
}

void matmul(size_t M, size_t N, size_t K, float A[M][K], float B[K][N], float ret[M][N]) {
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      ret[m][n] = 0.0f;
      for (size_t k = 0; k < K; k++) {
        ret[m][n] += A[m][k] * B[k][n];
      }
    }
  }
}

float op_relu(float x) { return x > 0.0f ? x : 0.0f; }
float op_drelu(float g, float x) { return x > 0.0f ? g : 0.0f; }

void unary_elemwise(size_t M, size_t N, float A[M][N], float B[M][N], float (*op)(float)) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      B[i][j] = op(A[i][j]);
    }
  }
}

void binary_elemwise(size_t M, size_t N, float A[M][N], float B[M][N], float C[M][N], float (*op)(float, float)) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      C[i][j] = op(A[i][j], B[i][j]);
    }
  }
}

void transpose(size_t M, size_t N, float A[M][N], float B[N][M]) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      B[j][i] = A[i][j];
    }
  }
}

void softmax(size_t M, size_t N, float A[M][N], float B[M][N]) {
  float expCache[N];
  for (size_t i = 0; i < M; i++) {
    float expSum = 0.0f;
    for (size_t j = 0; j < N; j++) {
      expCache[j] = expf(A[i][j]);
      expSum += expCache[j];
    }
    for (size_t j = 0; j < OUT_DIM; j++) {
      B[i][j] = expCache[j] / expSum;
    }
  }
}

void train_step(params_t* params, float feats[][IN_DIM], float labels[][OUT_DIM], size_t start) {
  /* forward */
  float h1[BS][HIDDEN_DIM];
  matmul(BS, HIDDEN_DIM, IN_DIM, &feats[start], params->w1, h1);
  for (size_t i = 0; i < BS; i++) {
    for (size_t j = 0; j < HIDDEN_DIM; j++) {
      h1[i][j] += params->b1[j];
    }
  }

  float a1[BS][HIDDEN_DIM];
  unary_elemwise(BS, HIDDEN_DIM, h1, a1, op_relu);

  float h2[BS][OUT_DIM];
  matmul(BS, OUT_DIM, HIDDEN_DIM, a1, params->w2, h2);
  for (size_t i = 0; i < BS; i++) {
    for (size_t j = 0; j < OUT_DIM; j++) {
      h2[i][j] += params->b2[j];
    }
  }

  float a2[BS][OUT_DIM];
  softmax(BS, OUT_DIM, h2, a2);

  /* NLL loss */
  /*
  float loss = 0.0f;
  for (size_t i = 0; i < BS; i++) {
    float nll = 0.0f;
    for (size_t j = 0; j < OUT_DIM; j++) {
      nll += a2[i][j] * labels[start + i][j];
    }
    loss += -logf(nll);
  }
  loss /= BS;
  printf("loss: %f\n", loss);
  */

  /* backward */
  float d_h2[BS][OUT_DIM];
  for (size_t i = 0; i < BS; i++) {
    for (size_t j = 0; j < OUT_DIM; j++) {
      d_h2[i][j] = a2[i][j] - labels[start + i][j];
    }
  }

  float a1_t[HIDDEN_DIM][BS];
  transpose(BS, HIDDEN_DIM, a1, a1_t);
  float d_w2[HIDDEN_DIM][OUT_DIM];
  matmul(HIDDEN_DIM, OUT_DIM, BS, a1_t, d_h2, d_w2);
  float d_b2[OUT_DIM] = {0.0f};
  for (size_t i = 0; i < BS; i++) {
    for (size_t j = 0; j < OUT_DIM; j++) {
      d_b2[j] += d_h2[i][j];
    }
  }

  float w2_t[OUT_DIM][HIDDEN_DIM];
  transpose(HIDDEN_DIM, OUT_DIM, params->w2, w2_t);
  float d_a1[BS][HIDDEN_DIM];
  matmul(BS, HIDDEN_DIM, OUT_DIM, d_h2, w2_t, d_a1);

  float d_h1[BS][HIDDEN_DIM];
  binary_elemwise(BS, HIDDEN_DIM, d_a1, h1, d_h1, op_drelu);

  float x_t[IN_DIM][BS];
  transpose(BS, IN_DIM, &feats[start], x_t);

  float d_w1[IN_DIM][HIDDEN_DIM];
  matmul(IN_DIM, HIDDEN_DIM, BS, x_t, d_h1, d_w1);

  float d_b1[HIDDEN_DIM] = {0.0f};
  for (size_t i = 0; i < BS; i++) {
    for (size_t j = 0; j < HIDDEN_DIM; j++) {
      d_b1[j] += d_h1[i][j];
    }
  }

  /* gradient descent */
  for (size_t i = 0; i < IN_DIM; i++) {
    for (size_t j = 0; j < HIDDEN_DIM; j++) {
      params->w1[i][j] -= LR * d_w1[i][j];
    }
  }
  for (size_t i = 0; i < HIDDEN_DIM; i++) {
    params->b1[i] -= LR * d_b1[i];
  }
  for (size_t i = 0; i < HIDDEN_DIM; i++) {
    for (size_t j = 0; j < OUT_DIM; j++) {
      params->w2[i][j] -= LR * d_w2[i][j];
    }
  }
  for (size_t i = 0; i < OUT_DIM; i++) {
    params->b2[i] -= LR * d_b2[i];
  }
}

void evaluate(params_t* params, float feats[][IN_DIM], float labels[][OUT_DIM], size_t start, size_t test_size) {
  /* forward */
  float h1[test_size][HIDDEN_DIM];
  matmul(BS, HIDDEN_DIM, IN_DIM, &feats[start], params->w1, h1);
  for (size_t i = 0; i < test_size; i++) {
    for (size_t j = 0; j < HIDDEN_DIM; j++) {
      h1[i][j] += params->b1[j];
    }
  }

  float a1[test_size][HIDDEN_DIM];
  unary_elemwise(test_size, HIDDEN_DIM, h1, a1, op_relu);
  float h2[test_size][OUT_DIM];
  matmul(test_size, OUT_DIM, HIDDEN_DIM, a1, params->w2, h2);
  for (size_t i = 0; i < test_size; i++) {
    for (size_t j = 0; j < OUT_DIM; j++) {
      h2[i][j] += params->b2[j];
    }
  }
  float a2[test_size][OUT_DIM];
  softmax(test_size, OUT_DIM, h2, a2);

  /* precision */
  size_t count = 0;
  for (size_t i = 0; i < test_size; i++) {
    size_t pred_label, true_label;
    float max_prob = -INFINITY;
    for (size_t j = 0; j < OUT_DIM; j++) {
      if (a2[i][j] > max_prob) {
        max_prob = a2[i][j];
        pred_label = j;
      }
      if (labels[start + i][j] == 1.0f) {
        true_label = j;
      }
    }
    if (pred_label == true_label) {
      count += 1;
    }
  }
  float precision = (float)count / test_size;


  float loss = 0.0f;
  for (size_t i = 0; i < test_size; i++) {
    float nll = 0.0f;
    for (size_t j = 0; j < OUT_DIM; j++) {
      nll += a2[i][j] * labels[start + i][j];
    }
    loss += -logf(nll);
  }
  loss /= test_size;
  printf("precision=%.6f, loss=%.6f\n", precision, loss);
}

void parse_example(char* line, float* feat, float* label) {
  const char *tok;
  size_t i = 0;
  for (tok = strtok(line, ","); tok; tok=strtok(NULL, ","), i++) {
    if (i < IN_DIM) {
      feat[i] = strtof(tok, NULL);
      continue;
    }
    if (strcmp(tok, "Iris-setosa") == 0) {
      label[0] = 1.0f;
    } else if (strcmp(tok, "Iris-versicolor") == 0) {
      label[1] = 1.0f;
    } else {
      label[2] = 1.0f;
    }
  }
}

void load_dataset(float feats[][IN_DIM], float labels[][OUT_DIM]) {
  FILE *fp = fopen(DATA_PATH, "r");
  char line[64];
  size_t i = 0;
  while (fgets(line, 64, fp)) {
    line[strcspn(line, "\n")] = '\0';    // get rid of the trailing \n
    if (strlen(line)) {
      char *dup = strdup(line);
      parse_example(dup, feats[i], labels[i]);
      free(dup);
      i++;
    }
  }
  fclose(fp);
}

void preprocess_dataset(float feats[][IN_DIM], float labels[][OUT_DIM]) {
  /* normalization */
  for (size_t j = 0; j < IN_DIM; j++) {
    float mean = 0.0f, std = 0.0f;
    for (size_t i = 0; i < N_EXAMPLES; i++) {
      mean += feats[i][j];
    }
    mean /= N_EXAMPLES;
    for (size_t i = 0; i < N_EXAMPLES; i++) {
      std += powf(feats[i][j] - mean, 2);
    }
    std = sqrtf(std / N_EXAMPLES);
    // printf("dim %zu mean: %f std: %f\n", j, mean, std);

    for (size_t i = 0; i < N_EXAMPLES; i++) {
      feats[i][j] = (feats[i][j] - mean) / std;
    }
  }

  /* shuffle dataset */
  FILE *fp = fopen(SPLIT_PATH, "r");
  char line[512];
  int random_indices[N_EXAMPLES];
  int count = 0;

  // Read indices
  if (fgets(line, sizeof(line), fp)) {
    char *token = strtok(line, ",");
    while (token) {
      random_indices[count++] = atoi(token);
      token = strtok(NULL, ",");
    }
  }

  if (fgets(line, sizeof(line), fp)) {
    char *token = strtok(line, ",");
    while (token) {
      random_indices[count++] = atoi(token);
      token = strtok(NULL, ",");
    }
  }
  fclose(fp);

  // temporary arrays to hold shuffled data
  float t_feats[N_EXAMPLES][IN_DIM];
  float t_labels[N_EXAMPLES][OUT_DIM];
  for (size_t i = 0; i < N_EXAMPLES; i++) {
    size_t j = random_indices[i];
    memcpy(t_feats[i], feats[j], sizeof(float) * IN_DIM);
    memcpy(t_labels[i], labels[j], sizeof(float) * OUT_DIM);
  }
  // shuffled data back to original arrays
  memcpy(feats, t_feats, sizeof(float) * N_EXAMPLES * IN_DIM);
  memcpy(labels, t_labels, sizeof(float) * N_EXAMPLES * OUT_DIM);
}

int main() {
  /* timing */
  clock_t start = clock();
  /* load data */
  int train_size = (int)(N_EXAMPLES * 0.8f);
  float feats[N_EXAMPLES][IN_DIM];
  float labels[N_EXAMPLES][OUT_DIM] = {{0.0f}};

  load_dataset(feats, labels);
  preprocess_dataset(feats, labels);

  params_t params;
  // initialize_params(&params);
  load_params(&params);
  for (int epoch = 0; epoch < N_EPOCH; epoch++) {
    for (int step = 0; step < train_size / BS; step++) {
      train_step(&params, feats, labels, step * BS);
    }
  }
  evaluate(&params, feats, labels, train_size, N_EXAMPLES - train_size);
  clock_t end = clock();
  printf("time=%.4fms\n", (float)(end - start) / CLOCKS_PER_SEC * 1000);
}
