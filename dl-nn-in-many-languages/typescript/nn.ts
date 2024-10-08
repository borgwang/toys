// ts-node run.ts
import * as fs from 'fs'

function integerRandom(l: number, r: number): number {
  return Math.floor(uniformRandom(l, r))
}
function uniformRandom(l: number = 0, r: number = 1): number {
  return (Math.random() - l) * (r - l)
}
function gaussianRandom(mu: number = 0, sigma: number = 1): number {
  let u1 = uniformRandom(), u2 = uniformRandom()
  let z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2)
  return z0 * sigma + mu
}

function newArray(m: number, n: number, generator: () => number): number[][] {
  let ret: number[][] = []
  for (let i = 0; i < m; i++) {
    ret[i] = []
    for (let j = 0; j < n; j++) {
      ret[i][j] = generator()
    }
  }
  return ret
}

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
  let bound1 = Math.sqrt(6.0 / (inDim + hiddenDim))
  let bound2 = Math.sqrt(6.0 / (hiddenDim + outDim))
  return {
    w1: newArray(inDim, hiddenDim, () => uniformRandom(-bound1, bound1)),
    b1: newArray(1, hiddenDim, () => 0.0),
    w2: newArray(hiddenDim, outDim, () => uniformRandom(-bound2, bound2)),
    b2: newArray(1, outDim, () => 0.0)
  }
}

function readCsv(path: string): string[][] {
  let data = fs.readFileSync(path, "utf-8")
  let ret = data.split("\n")
    .filter(line => line && line.length > 0)
    .map(line => line.split(","))
  return ret
}

function prepareDataset(): {train: Dataset, test: Dataset} {
  let data: string[][] = readCsv("../data/iris.data")
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
  // random shuffle (Fisher-Yates)
  let numSamples: number = labels.length
  let shuffleFeats: number[][] = [], shuffleLabels: number[][] = []
  for (let i = 0; i < numSamples; i++) {
    let idx: number = integerRandom(0, feats.length)
    shuffleFeats.push(feats[idx])
    shuffleLabels.push(labels[idx])
    feats.splice(idx, 1)
    labels.splice(idx, 1)
  }
  feats = shuffleFeats
  labels = shuffleLabels

  // split train/test set
  let numTrain = Math.floor(numSamples * 0.8)
  let trainX = feats.filter((_, idx) => idx < numTrain)
  let trainY = labels.filter((_, idx) => idx < numTrain)
  let testX = feats.filter((_, idx) => idx >= numTrain)
  let testY = labels.filter((_, idx) => idx >= numTrain)

  // mean and std from trainX
  let meanX: number[] = [], stdX: number[] = []
  for (let i = 0; i < numColumns - 1; i++) {
    let col = trainX.map(feat => feat[i])
    let mean = col.reduce((a, b) => (a + b)) / numTrain
    let variance = col.map(f => Math.pow(f - mean, 2)).reduce((a, b) => (a + b)) / numTrain
    let std = Math.sqrt(variance)
    meanX.push(mean)
    stdX.push(std)
  }
  console.log(`mean: ${meanX}`)
  console.log(`std: ${stdX}`)
  // normalize trainX and testX with mean and std
  trainX = trainX.map(feat => feat.map((f, i) => (f - meanX[i]) / stdX[i]))
  testX = testX.map(feat => feat.map((f, i) => (f - meanX[i]) / stdX[i]))
  return { train: { x: trainX, y: trainY }, test: { x: testX, y: testY } }
}

function getShape(input: number[][]): number[] {
  return [input.length, input[0].length]
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

function train(params: Params, dataset: Dataset) {
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
  let loss = getNLLLoss(probs, dataset.y)
  //console.log("loss: " + loss)

  // backward
  let dLogits = mElemwise((a, b) => (a - b), probs, dataset.y)
  let dw2 = mMatMul(mTranspose(a1), dLogits)
  // tricky way to sum along the second axis
  let db2 = mTranspose(mTranspose(dLogits).map(x => [x.reduce((a, b) => (a + b))]))

  let da1 = mMatMul(dLogits, mTranspose(params.w2))
  let dh1 = mElemwise(x => x > 0 ? 1 : 0, da1)
  let dw1 = mMatMul(mTranspose(dataset.x), dh1)
  let db1 = mTranspose(mTranspose(dh1).map(x => [x.reduce((a, b) => (a + b))]))
  // gradient descent
  params.w1 = mElemwise((a, b) => (a - leanringRate * b), params.w1, dw1)
  params.b1 = mElemwise((a, b) => (a - leanringRate * b), params.b1, db1)
  params.w2 = mElemwise((a, b) => (a - leanringRate * b), params.w2, dw2)
  params.b2 = mElemwise((a, b) => (a - leanringRate * b), params.b2, db2)
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
  console.log(`precisionn: ${precision}`)
}

// training
let params = getInitParams(4, 3, 20)
let dataset = prepareDataset()
let batchSize = 30
let trainSize = dataset.train.x.length

const leanringRate = 0.0003
const numEpochs = 1000
//const numEpochs = 1
for (let ep = 0; ep < numEpochs; ep++) {
  for (let b = 0; b < trainSize / batchSize; b++) {
    let batch = {
      x: dataset.train.x.filter((_, idx) => (idx >= b*batchSize) && (idx < (b+1)*batchSize)),
      y: dataset.train.y.filter((_, idx) => (idx >= b*batchSize) && (idx < (b+1)*batchSize))
    }
    train(params, batch)
  }
  if (ep % 100 == 0) {
    console.log(`epoch ${ep}`)
    evaluate(params, dataset.test)
  }
}
