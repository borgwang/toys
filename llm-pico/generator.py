import numpy as np
from utils import sample, softmax


class Generator:

  def __init__(self, model, tokenizer, max_tokens=None, t=1.0, stream=True):
    self.model = model
    self.tokenizer = tokenizer

    self.max_tokens = model.ctx_size if not max_tokens else max_tokens
    self.t = t

    self.stream = stream
    self.printed_len = 0

  def __call__(self, inputs):
    start_ids = self.tokenizer.encode(inputs)

    gen_ids = []
    cnt = len(start_ids)
    while cnt < self.max_tokens:
      ids_cond = (start_ids + gen_ids)
      logits = self.model.forward(ids_cond)
      out_id = int(np.argmax(logits)) if self.t == 0 else sample(softmax(logits / (self.t + 1e-8)))
      gen_ids.append(out_id)
      cnt += 1
      if self.stream:
        self.streamprint(start_ids + gen_ids)

    if self.stream:
      print()
    return gen_ids, self.tokenizer.decode(gen_ids)

  def streamprint(self, ids):
    decoded = self.tokenizer.decode(ids)
    print(decoded[self.printed_len:], end="", flush=True)
    self.printed_len = len(decoded)
