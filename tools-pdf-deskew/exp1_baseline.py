import argparse
import os

import numpy as np
from evaluate import evaluate
from pdf2image import convert_from_path

DPI = 200

def process(img):
  w, h = img.size
  max_std = 0
  best_degree = None
  for degree in np.arange(-1.5, 1.5 + 1e-4, 0.1):
    img2 = img.rotate(degree, expand=True, fillcolor="white")
    offset_x, offset_y = w // 8, h // 8
    img3a = np.array(img2)[offset_y:-offset_y, offset_x:-offset_x]
    std = np.std(img3a.mean(axis=1))
    if std > max_std:
      max_std = std
      best_degree = degree
  tarr = np.array(img.rotate(best_degree, fillcolor="white"))
  return tarr


def main():
  assert os.path.exists(args.input)
  last_page = None if args.n_pages is None else args.first_page + args.n_pages - 1

  imgs = convert_from_path(args.input, thread_count=8, first_page=args.first_page,
                           last_page=last_page, grayscale=True, dpi=DPI)
  print(f"[INFO] {len(imgs)} pages in total. start from page {args.first_page}")

  arrs = [np.array(img) for img in imgs]
  newarrs = [process(img) for img in imgs]
  scores = [evaluate(arr, newarr, i+args.first_page) for i, (arr, newarr) in enumerate(zip(arrs, newarrs))]
  print(f"[INFO] mean score: {100*np.mean(scores):.2f}%")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=str, required=True)
  parser.add_argument("--first_page", type=int, default=1)
  parser.add_argument("--n_pages", type=int, default=None)
  args = parser.parse_args()
  main()
