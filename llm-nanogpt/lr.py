import math

class LRCosineAnnealing:
  def __init__(self, lr, min_lr=3e-5, warmup_iters=0, decay_iters=1000):
    self.lr = lr
    self.min_lr = min_lr
    self.warmup_iters = warmup_iters
    self.decay_iters = decay_iters

  def __call__(self, i):
    if i < self.warmup_iters:
      return self.lr * i / self.warmup_iters
    if i > self.decay_iters:
      return self.min_lr

    decay_ratio = (i - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return self.min_lr + coeff * (self.lr - self.min_lr)
