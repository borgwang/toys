// ts-node run.ts
import * as fs from 'fs'

const LR = 0.0003
const N_EPOCH = 1000
const BS = 30
const IN_DIM = 4
const OUT_DIM = 3
const HIDDEN_DIM = 20

const PARAMS_PATH = "data/params.txt"
const DATA_PATH = "data/iris.data"
const SPLIT_PATH = "data/split.txt"

function mSum(A: number[][]): number {
  let ret = 0
  for (let i = 0; i < A.length; i++) {
    ret += A[i].reduce((a, b) => (a + b))
  }
  return ret
}
function mMean(A: number[][]): number {
  return mSum(A) / (A.length * A[0].length)
}
function mElemwise(op: (...args: number[]) => number, ...inputs: number[][][]): number[][] {
  let ret: number[][] = []
  for (let i = 0; i < inputs[0].length; i++) {
    ret[i] = []
    for (let j = 0; j < inputs[0][0].length; j++) {
      ret[i][j] = op(...inputs.map(x => x[i][j]))
    }
  }
  return ret
}
function mMatMul(A: number[][], B: number[][]): number[][] {
  let ret: number[][] = []
  let M = A.length, K = A[0].length, N = B[0].length
  for (let m = 0; m < M; m++) {
    ret[m] = []
    for (let n = 0; n < N; n++) {
      let val = 0
      for (let k = 0; k < K; k++) {
        val += A[m][k] * B[k][n]
      }
      ret[m][n] = val
    }
  }
  return ret
}

function mTranspose(A: number[][]): number[][] {
  let M = A.length, N = A[0].length
  let ret: number[][] = Array.from(Array(N), () => new Array(M))
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      ret[j][i] = A[i][j]
    }
  }
  return ret
}

// weight initialize
interface Params {
  w1: number[][]
  b1: number[][]
  w2: number[][]
  b2: number[][]
}

interface Dataset {
  x: number[][]
  y: number[][]
}

function getInitParams(inDim: number, outDim: number, hiddenDim: number): Params {
  const data = fs.readFileSync(PARAMS_PATH, 'utf-8');
  const lines = data.trim().split('\n');

  const w1 = lines[0].split(',').map(Number);
  const b1 = lines[1].split(',').map(Number);
  const w2 = lines[2].split(',').map(Number);
  const b2 = lines[3].split(',').map(Number);

  return {
    w1: reshape(w1, inDim, hiddenDim),
    b1: [b1],
    w2: reshape(w2, hiddenDim, outDim),
    b2: [b2]
  };
}

function reshape(arr: number[], rows: number, cols: number): number[][] {
  const result: number[][] = [];
  for (let i = 0; i < rows; i++) {
    result.push(arr.slice(i * cols, (i + 1) * cols));
  }
  return result;
}

function readCsv(path: string): string[][] {
  let data = fs.readFileSync(path, "utf-8")
  let ret = data.split("\n")
    .filter((line: string) => line && line.length > 0)
    .map((line: string) => line.split(","))
  return ret
}

function prepareDataset(): {train: Dataset, test: Dataset} {
  let data: string[][] = readCsv(DATA_PATH)
  let labels: number[][] = []
  let feats: number[][] = []
  let numColumns = data[0].length
  for (let i = 0; i < data.length; i++) {
    feats[i] = []
    // one-hot label
    switch (data[i][numColumns - 1]) {
      case "Iris-setosa":
        labels[i] = [1, 0, 0]; break
      case "Iris-versicolor":
        labels[i] = [0, 1, 0]; break
      case "Iris-virginica":
        labels[i] = [0, 0, 1]; break
      default: break
    }
    for (let j = 0; j < numColumns - 1; j++) {
      feats[i][j] = parseFloat(data[i][j])
    }
  }

  let numSamples: number = labels.length
  // normalize feats
  let meanX: number[] = [], stdX: number[] = []
  for (let i = 0; i < numColumns - 1; i++) {
    let col = feats.map(feat => feat[i])
    let mean = col.reduce((a, b) => (a + b)) / numSamples
    let variance = col.map(f => Math.pow(f - mean, 2)).reduce((a, b) => (a + b)) / numSamples
    let std = Math.sqrt(variance)
    meanX.push(mean)
    stdX.push(std)
  }
  feats = feats.map(feat => feat.map((f, i) => (f - meanX[i]) / stdX[i]))

  // Read random indices from split.txt
  let indices = fs.readFileSync(SPLIT_PATH, 'utf-8').trim().split('\n').map(line => line.split(",").map(Number));
  let trainIndices = indices[0]
  let testIndices = indices[1]
  // Use the indices to split the dataset
  let trainX = trainIndices.map((index: number) => feats[index]);
  let trainY = trainIndices.map((index: number) => labels[index]);
  let testX = testIndices.map((index: number) => feats[index]);
  let testY = testIndices.map((index: number) => labels[index]);
  return { train: { x: trainX, y: trainY }, test: { x: testX, y: testY } }
}

function softmax(logit: number[][]): number[][] {
  let exp = mElemwise(x => Math.exp(x), logit)
  let expSum = exp.map(e => e.reduce((a, b) => (a + b)))
  let prob = exp.map((e, i) => e.map(a => a / expSum[i]))
  return prob
}

function getNLLLoss(probs: number[][], labels: number[][]): number {
  let liklihood =  mElemwise((a, b) => (a * b), probs, labels)
    .map(row => [row.reduce((a, b) => a + b)])  // (B, 1)
  let nll = mElemwise(x => -Math.log(x), liklihood)
  return mMean(nll)
}

function argMax(array: number[]): number {
  return array.map((num, idx) => [idx, num]).reduce((a, b) => (a[1] > b[1] ? a : b))[0]
}

function train_step(params: Params, dataset: Dataset) {
  let numSamples = dataset.x.length
  // forward
  let h1 = mMatMul(dataset.x, params.w1)
  let b1 = Array(numSamples).fill(params.b1[0])
  h1 = mElemwise((a, b) => (a + b), h1, b1)
  let a1 = mElemwise(x => x > 0 ? x : 0, h1)

  let h2 = mMatMul(a1, params.w2)
  let b2 = Array(numSamples).fill(params.b2[0])
  h2 = mElemwise((a, b) => (a + b), h2, b2)


  let probs = softmax(h2)
  //console.log("loss: " + getNLLLoss(probs, dataset.y))

  // backward
  let dh2 = mElemwise((a, b) => (a - b), probs, dataset.y)
  let dw2 = mMatMul(mTranspose(a1), dh2)
  // tricky way to sum along the second axis
  let db2 = mTranspose(mTranspose(dh2).map(x => [x.reduce((a, b) => (a + b))]))

  let da1 = mMatMul(dh2, mTranspose(params.w2))
  let dh1 = mElemwise((g, x) => x > 0 ? g : 0, da1, h1)
  let dw1 = mMatMul(mTranspose(dataset.x), dh1)
  let db1 = mTranspose(mTranspose(dh1).map(x => [x.reduce((a, b) => (a + b))]))
  // gradient descent
  params.w1 = mElemwise((a, b) => (a - LR * b), params.w1, dw1)
  params.b1 = mElemwise((a, b) => (a - LR * b), params.b1, db1)
  params.w2 = mElemwise((a, b) => (a - LR * b), params.w2, dw2)
  params.b2 = mElemwise((a, b) => (a - LR * b), params.b2, db2)
}

function evaluate(params: Params, dataset: Dataset) {
  let numSamples = dataset.x.length
  // forward
  let h1 = mMatMul(dataset.x, params.w1)
  let b1 = Array(numSamples).fill(params.b1[0])
  h1 = mElemwise((a, b) => (a + b), h1, b1)

  let a1 = mElemwise(x => x > 0 ? x : 0, h1)
  let h2 = mMatMul(a1, params.w2)
  let b2 = Array(numSamples).fill(params.b2[0])
  let logits = mElemwise((a, b) => (a + b), h2, b2)
  let probs = softmax(logits)
  let predLabels = probs.map(argMax)
  let trueLabels = dataset.y.map(argMax)
  let precision: number = predLabels
      .map((pred, i) => Number(pred === trueLabels[i]))
      .reduce((a, b) => (a + b)) / numSamples
  let loss = getNLLLoss(probs, dataset.y)
  console.log(`precision=${precision.toFixed(6)}, loss=${loss.toFixed(6)}`)
}

function main() {
  // timing
  const startTime = performance.now();

  // training
  let params = getInitParams(IN_DIM, OUT_DIM, HIDDEN_DIM);
  let dataset = prepareDataset();
  let trainSize = dataset.train.x.length;

  for (let ep = 0; ep < N_EPOCH; ep++) {
    for (let b = 0; b < trainSize / BS; b++) {
      let batch = {
        x: dataset.train.x.filter((_, idx) => (idx >= b*BS) && (idx < (b+1)*BS)),
        y: dataset.train.y.filter((_, idx) => (idx >= b*BS) && (idx < (b+1)*BS))
      };
      train_step(params, batch);
    }
  }
  evaluate(params, dataset.test);

  // timing
  const endTime = performance.now();
  console.log(`time=${(endTime - startTime).toFixed(4)} ms`);
}

main();
