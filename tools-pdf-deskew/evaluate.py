import numpy as np


def evaluate(arr, newarr, index=None, verbose=False):
  assert arr.shape == newarr.shape
  horizontal_boost = (np.var(newarr.mean(1)) - np.var(arr.mean(1))) / np.var(arr.mean(1))
  vertical_boost = (np.var(newarr.mean(0)) - np.var(arr.mean(0))) / np.var(arr.mean(0))
  score = (horizontal_boost + vertical_boost) / 2
  if verbose:
    print(f"horizontal: {horizontal_boost:.4f}")
    print(f"vertical: {vertical_boost:.4f}")
    if index:
      print(f"page_idx={index}")
    print(f"score={score:.4f}")
  return score
