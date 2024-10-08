use std::error::Error;
use std::convert::TryInto;
use csv::ReaderBuilder;
use rand::Rng;
use std::time::Instant;

const N_EPOCH: usize = 1000;
const N_EXAMPLES: usize = 150;
const IN_DIM: usize = 4;
const OUT_DIM: usize = 3;
const HIDDEN_DIM: usize = 20;
const BS: usize = 30;
const LR: f32 = 0.0003;


fn random_uniform(min:f32, max:f32) -> f32 {
  let mut rng = rand::thread_rng();
  rng.gen_range(min..max)
}

#[derive(Debug)]
struct Params {
  w1: [[f32; HIDDEN_DIM]; IN_DIM],
  b1: [f32; HIDDEN_DIM],
  w2: [[f32; OUT_DIM]; HIDDEN_DIM],
  b2: [f32; OUT_DIM],
}

#[derive(Debug)]
struct Dataset {
  x: Vec<[f32; IN_DIM]>,
  y: Vec<[f32; OUT_DIM]>,
  _curr_idx: usize,
}

impl Dataset {
  pub fn new(x: Vec<[f32; IN_DIM]>, y: Vec<[f32; OUT_DIM]>) -> Self {
    Dataset { x, y, _curr_idx: 0}
  }

  fn shuffle(&mut self) {
    let mut rng = rand::thread_rng();
    for i in (1..N_EXAMPLES).rev() {
      let a = rng.gen_range(0..i);
      self.x.swap(a, i);
      self.y.swap(a, i);
    }
  }

  fn normalize(&mut self) {
    for j in 0..IN_DIM {
      // mean & std
      let mean: f32 = self.x.iter().map(|x| x[j]).sum::<f32>() / N_EXAMPLES as f32;
      let std: f32 = (self.x.iter().map(|x| (x[j] - mean).powi(2)).sum::<f32>() / N_EXAMPLES as f32).sqrt();
      println!("dim {} mean: {} std: {}", j, mean, std);

      // Normalize the data
      for i in 0..N_EXAMPLES {
        self.x[i][j] = if std != 0.0 { (self.x[i][j] - mean) / std } else { 0.0 };
      }
    }
  }

  fn split(&self, ratio: f32) -> (Dataset, Dataset) {
    let split: usize = (ratio * self.x.len() as f32) as usize;
    let train_set = Dataset::new(self.x[..split].to_vec(), self.y[..split].to_vec());
    let test_set = Dataset::new(self.x[split..].to_vec(), self.y[split..].to_vec());
    (train_set, test_set)
  }

  fn size(&self) -> usize {
    self.x.len()
  }

  fn next_batch(&mut self) -> ([[f32; IN_DIM]; BS], [[f32; OUT_DIM]; BS]) {
    if self._curr_idx + BS > self.size() {
      self._curr_idx = 0;
    }
    let new_idx = self._curr_idx + BS;
    let x = self.x[self._curr_idx..new_idx].try_into().unwrap();
    let y = self.y[self._curr_idx..new_idx].try_into().unwrap();
    self._curr_idx = new_idx;
    (x, y)
  }

}

impl Params {
    fn new() -> Params {
      let mut params = Params {
        w1: [[0.0; HIDDEN_DIM]; IN_DIM],
        b1: [0.0; HIDDEN_DIM],
        w2: [[0.0; OUT_DIM]; HIDDEN_DIM],
        b2: [0.0; OUT_DIM],
      };
      let bound1 = (6.0 / (IN_DIM as f32 + HIDDEN_DIM as f32)).sqrt();
      for j in 0..HIDDEN_DIM {
        for i in 0..IN_DIM {
          params.w1[i][j] = random_uniform(-bound1, bound1);
        }
      }
      let bound2 = (6.0 / (HIDDEN_DIM as f32 + OUT_DIM as f32)).sqrt();
      for j in 0..OUT_DIM {
        for i in 0..HIDDEN_DIM {
          params.w2[i][j] = random_uniform(-bound2, bound2);
        }
      }
      params
    }
}

fn load_dataset(path: &str, x_vec: &mut Vec<[f32; IN_DIM]>, y_vec: &mut Vec<[f32; OUT_DIM]>) -> Result<(), Box<dyn Error>> {
  let mut rdr = ReaderBuilder::new().has_headers(false).from_path(path)?;
  for result in rdr.records() {
    let record = result?;
    let feats_vec: Vec<f32> = record.iter().take(IN_DIM).map(|s| s.parse::<f32>().unwrap()).collect();
    let x: [f32; IN_DIM] = feats_vec.try_into().expect("Incorrect length for feat array");
    let label: &str = &record[4];
    let mut y: [f32; OUT_DIM] = [0.0; OUT_DIM];
    match label {
      "Iris-setosa" => y[0] = 1.0,
      "Iris-versicolor" => y[1] = 1.0,
      "Iris-virginica" => y[2] = 1.0,
      _ => panic!("Invalid label value '{}'", label),
    }
    x_vec.push(x);
    y_vec.push(y);
  }
  Ok(())
}

fn matmul<const M: usize, const N: usize, const K: usize>(
  a: &[[f32; K]; M], b: &[[f32; N]; K]
) -> [[f32; N]; M] {
  let mut ret:[[f32; N]; M] = [[0.0; N]; M];
  for i in 0..M {
    for j in 0..N {
      ret[i][j] = 0.0;
      for k in 0..K {
        ret[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  ret
}

fn softmax<const M: usize, const N: usize> (input: &[[f32; N]; M]) -> [[f32; N]; M] {
  let mut ret: [[f32; N]; M] = [[0.0; N]; M];
  let mut exp_cache: [f32; N] = [0.0; N];
  for i in 0..M {
    let mut exp_sum: f32 = 0.0;
    for j in 0..N {
      exp_cache[j] = input[i][j].exp();
      exp_sum += exp_cache[j];
    }
    for j in 0..N {
      ret[i][j] = exp_cache[j] / exp_sum;
    }
  }
  ret
}

fn unary_elemwise<F, const M: usize, const N: usize> (
  input: &[[f32; N]; M], op: F,
) -> [[f32; N]; M] where F: Fn(f32) -> f32 {
  let mut ret = [[0.0; N]; M];
  for i in 0..M {
    for j in 0..N {
      ret[i][j] = op(input[i][j]);
    }
  }
  ret
}

fn binary_elemwise<F, const M: usize, const N: usize> (
  a: &[[f32; N]; M], b: &[[f32; N]; M], op: F,
) -> [[f32; N]; M] where F: Fn(f32, f32) -> f32 {
  let mut ret = [[0.0; N]; M];
  for i in 0..M {
    for j in 0..N {
      ret[i][j] = op(a[i][j], b[i][j]);
    }
  }
  ret
}

fn transpose<const M: usize, const N: usize> (
  input: &[[f32; N]; M]
) -> [[f32; M]; N] {
  let mut ret = [[0.0; M]; N];
  for i in 0..M {
    for j in 0..N {
      ret[j][i] = input[i][j];
    }
  }
  ret
}

fn train_step(params: &mut Params, x: &[[f32; IN_DIM]; BS], y: &[[f32; OUT_DIM]; BS]) {
  // Forward pass
  let mut h1 = matmul(x, &params.w1);
  for i in 0..BS {
    for j in 0..HIDDEN_DIM {
      h1[i][j] += params.b1[j];
    }
  }
  let a1 = unary_elemwise(&h1, |x| if x > 0.0 {x} else {0.0});

  let mut h2 = matmul(&a1, &params.w2);
  for i in 0..BS {
    for j in 0..OUT_DIM {
      h2[i][j] += params.b2[j];
    }
  }
  let a2 = softmax(&mut h2);

  // // Loss calculation: Cross-entropy loss
  // let mut loss: f32 = 0.0;
  // for i in 0..BS {
  //   let mut nll: f32 = 0.0;
  //   for j in 0..OUT_DIM {
  //     nll += a2[i][j] * y[i][j];
  //   }
  //   loss += -nll.ln();
  // }
  // loss /= BS as f32;
  // println!("loss: {}", loss);

  // Backward pass
  // d_h2 = a2 - y (derivative of softmax with cross-entropy)
  let d_h2 = binary_elemwise(&a2, &y, |a, b| a - b);
  // d_w2 = a1^T * d_h2
  let d_w2 = matmul(&transpose(&a1), &d_h2);
  // d_b2 = sum(d_h2, axis=0)
  let mut d_b2 = [0.0; OUT_DIM];
  for i in 0..BS {
    for j in 0..OUT_DIM {
      d_b2[j] += d_h2[i][j];
    }
  }

  // d_a1 = d_h2 * w2^T
  let d_a1 = matmul(&d_h2, &transpose(&params.w2));
  // d_h1 = d_a1 * ReLU'(h1)
  let d_h1 = binary_elemwise(&d_a1, &h1, |a, b| if b > 0.0 {a} else {0.0});

  // d_w1 = x^T * d_h1
  let d_w1 = matmul(&transpose(x), &d_h1);
  // d_b1 = sum(d_h1, axis=0)
  let mut d_b1 = [0.0; HIDDEN_DIM];
  for i in 0..BS {
    for j in 0..HIDDEN_DIM {
      d_b1[j] += d_h1[i][j];
    }
  }

  // Gradient descent update
  for i in 0..IN_DIM {
    for j in 0..HIDDEN_DIM {
      params.w1[i][j] -= LR * d_w1[i][j];
    }
  }
  for i in 0..HIDDEN_DIM {
    params.b1[i] -= LR * d_b1[i];
  }
  for i in 0..HIDDEN_DIM {
    for j in 0..OUT_DIM {
      params.w2[i][j] -= LR * d_w2[i][j];
    }
  }
  for i in 0..OUT_DIM {
    params.b2[i] -= LR * d_b2[i];
  }
}

fn evaluate(params: &Params, x: &[[f32; IN_DIM]; BS], y: &[[f32; OUT_DIM]; BS]) -> f32 {
  // forward pass and calcualte precision
  let mut h1 = matmul(x, &params.w1);
  for i in 0..BS {
    for j in 0..HIDDEN_DIM {
      h1[i][j] += params.b1[j];
    }
  }
  let a1 = unary_elemwise(&h1, |x| if x > 0.0 {x} else {0.0});
  let mut h2 = matmul(&a1, &params.w2);
  for i in 0..BS {
    for j in 0..OUT_DIM {
      h2[i][j] += params.b2[j];
    }
  }
  let a2 = softmax(&mut h2);
  let mut count = 0;
  for i in 0..BS {
    let mut max_prob = -f32::INFINITY;
    let (mut pred_label, mut true_label) = (0, 0);
    for j in 0..OUT_DIM {
      if a2[i][j] > max_prob {
        max_prob = a2[i][j];
        pred_label = j;
      }
      if y[i][j] == 1.0 {
        true_label = j;
      }
    }
    if pred_label == true_label {
      count += 1;
    }
  }
  let precision = count as f32 / BS as f32;
  precision
}

fn main() {
  // timing
  let st = Instant::now();

  let path: &str = "../data/iris.data";

  let mut x: Vec<[f32; IN_DIM]> = Vec::new();
  let mut y: Vec<[f32; OUT_DIM]> = Vec::new();

  let _ = load_dataset(&path, &mut x, &mut y);

  let mut dataset = Dataset::new(x, y);
  dataset.shuffle();
  dataset.normalize();
  let (mut train_set, test_set) = dataset.split(0.8);

  // init params
  let mut params = Params::new();

  // train & evaluate
  let test_x: [[f32; IN_DIM]; BS] = test_set.x.clone().try_into().unwrap();
  let test_y: [[f32; OUT_DIM]; BS] = test_set.y.clone().try_into().unwrap();

  for epoch in 0..N_EPOCH {
    for _step in 0..(train_set.size() / BS) {
      let (x, y) = train_set.next_batch();
      train_step(&mut params, &x, &y);
    }
    if epoch % 100 == 0 {
      let precision = evaluate(&params, &test_x, &test_y);
      println!("epoch: {}, precision: {}", epoch, precision);
    }
  }
  let duration = st.elapsed();
  println!("time cost: {:?}", duration);
}
