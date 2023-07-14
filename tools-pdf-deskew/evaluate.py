import numpy as np

def evaluate(arr, newarr, index=None, verbose=False):
  assert arr.shape == newarr.shape
  ratio = arr.shape[0] / arr.shape[1]
  horizontal_boost = (np.std(newarr.mean(1)) - np.std(arr.mean(1))) / np.std(arr.mean(1))
  vertical_boost = (np.std(newarr.mean(0)) - np.std(arr.mean(0))) / np.std(arr.mean(0))
  score = horizontal_boost + ratio * vertical_boost
  if verbose:
    #print(f"horizontal: {horizontal_boost:.4f}")
    #print(f"vertical: {vertical_boost:.4f}")
    if index:
      print(f"page_idx={index}")
    print(f"score={score:.4f}")
  return score
