import argparse
import multiprocessing
import os
import tempfile
import time
from collections import defaultdict
from functools import partial

import cv2
import img2pdf
import numpy as np
from evaluate import evaluate
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression
from tqdm.auto import tqdm

DEGREE_LB = -1.2
DEGREE_UB = 1.2 + 1e-8
DEGREE_STEP = 0.02

DETECT_PART_TOP_BOTTOM = 1/3
DETECT_PART_LEFT_RIGHT = 1/3

DPI = 200
PLOT = int(os.getenv("PLOT", "0"))
WORKERS = int(os.getenv("WORKERS", "8"))

def func_star(func, args):
  return func(*args)

def multiprocs(func, argslist, n_workers:int=4, progress:bool=False, ordered:bool=True, star:bool=False):
  try:
    with multiprocessing.Pool(n_workers) as pool:
      map_ = pool.imap if ordered else pool.imap_unordered
      func = partial(func_star, func) if star else func
      if progress:
        ret = []
        with tqdm(total=len(argslist), smoothing=0.01, dynamic_ncols=True) as pbar:
          for r in map_(func, argslist):
            pbar.update()
            ret.append(r)
      else:
        ret = list(map_(func, argslist))
    return ret
  except Exception as e:
    raise e

def process(arr):
  if arr.mean() > 255*0.95:
    return arr

  # step1: get edges and corner points
  edges = get_edges(arr)
  points = {
      "tl": get_intersection(edges["top"], edges["left"]),
      "tr": get_intersection(edges["top"], edges["right"]),
      "br": get_intersection(edges["bottom"], edges["right"]),
      "bl": get_intersection(edges["bottom"], edges["left"])
  }
  # step2: deskew
  newarr = transform(arr, points)

  if PLOT:
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    for edge in edges.values():
      draw.line(tuple(edge), fill="black", width=2)
    img.show()
    Image.fromarray(newarr).show()
  return newarr

def get_edges(arr):
  arr = cv2.medianBlur(arr, 5)
  h, w = arr.shape
  edges = {}

  for side in ("top", "bottom"):
    best = defaultdict(int)
    x_ = np.arange(w).astype(int)
    offsets = np.arange(int(h*DETECT_PART_TOP_BOTTOM)) if side == "top" else np.arange(-1, -int(h*DETECT_PART_TOP_BOTTOM), -1)
    for degree in np.arange(DEGREE_LB, DEGREE_UB, DEGREE_STEP):
      tan = np.tan(np.abs(degree) / 180 * np.pi)
      if side == "top":
        y_ = tan*(w-x_) if degree > 0 else tan*x_
      else:
        y_ = h-tan*x_ if degree > 0 else h-tan*(w-x_)
      y_ = y_.astype(int)
      #concentration = np.array([np.mean(arr[y_ + offset, x_]) for offset in offsets])
      concentration = arr[y_.reshape((1, -1))+offsets.reshape((-1, 1)), np.tile(x_, (len(offsets), 1))].mean(1, dtype=np.float32)
      variance = ((concentration[1:] - concentration[:-1]) ** 2).sum()
      if variance >= best["var"]:
        best["var"] = variance
        best["concentration"] = concentration
        best["y_"] = y_
    best_offset = offsets[change_dectection(best["concentration"])]
    (y1, x1), (y2, x2) = (best["y_"][0] + best_offset, x_[0]), (best["y_"][-1] + best_offset, x_[-1])
    edges[side] = np.array([x1, y1, x2, y2])

  for side in ("left", "right"):
    best = defaultdict(int)
    y_ = np.arange(h).astype(int)
    offsets = np.arange(int(w*DETECT_PART_LEFT_RIGHT)) if side == "left" else np.arange(-1, -int(w*DETECT_PART_LEFT_RIGHT), -1)
    for degree in np.arange(DEGREE_LB, DEGREE_UB, DEGREE_STEP):
      tan = np.tan(np.abs(degree) / 180 * np.pi)
      if side == "left":
        x_ = tan*y_ if degree > 0 else tan*(h-y_)
      else:
        x_ = w-tan*(h-y_) if degree > 0 else w-tan*y_
      x_ = x_.astype(int)
      #concentration = np.array([np.mean(arr[y_, x_ + offset]) for offset in offsets])
      concentration = arr[np.tile(y_, (len(offsets), 1)), x_.reshape((1, -1))+offsets.reshape((-1, 1))].mean(1, dtype=np.float32)
      variance = np.var(concentration)
      if variance >= best["var"]:
        best["var"] = variance
        best["concentration"] = concentration
        best["x_"] = x_
    best_offset = offsets[change_dectection(best["concentration"])]
    (y1, x1), (y2, x2) = (y_[0], best["x_"][0] + best_offset), (y_[-1], best["x_"][-1] + best_offset)
    edges[side] = np.array([x1, y1, x2, y2])
  return edges

def get_intersection(line1, line2):

  def k_b(line):
    x1, y1, x2, y2 = line
    k = (y1 - y2) / (x1 - x2 + 1e-8)
    b = y1 - k * x1
    return k, b

  k_1, b_1 = k_b(line1)
  k_2, b_2 = k_b(line2)
  x = (b_1 - b_2) / (k_2 - k_1 + 1e-8)
  y = k_1 * x + b_1
  return np.array([x, y])

def transform(arr, points):

  def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

  oh, ow = arr.shape
  h1 = distance(points["tl"], points["bl"])
  h2 = distance(points["tr"], points["br"])
  th = max(int(h1), int(h2))
  w1 = distance(points["tl"], points["tr"])
  w2 = distance(points["bl"], points["br"])
  tw = max(int(w1), int(w2))

  dst = np.array([[0, 0], [tw-1, 0], [tw-1, th-1], [0, th-1]], dtype="float32")
  src = np.array(list(points.values()), dtype="float32")
  M = cv2.getPerspectiveTransform(src, dst)
  newarr = cv2.warpPerspective(arr, M, (tw, th))
  # padding
  pad_h, pad_w = (oh-th)//2, (ow-tw)//2
  return np.pad(newarr, ((pad_h, oh-th-pad_h), (pad_w, ow-tw-pad_w)), constant_values=255)

def change_dectection(points, step=1):
  points = np.array(points)[:, np.newaxis]
  xs = np.arange(len(points))[:, np.newaxis]
  best_err, best_sep = float("inf"), None
  model = LinearRegression()
  for sep in range(step, len(points), step):
    model.fit(xs[:sep], points[:sep])
    err1 = (np.abs(model.predict(xs[:sep]) - points[:sep]) ** 0.1).sum()
    model.fit(xs[sep:], points[sep:])
    err2 = (np.abs(model.predict(xs[sep:]) - points[sep:]) ** 0.1).sum()
    total_err = err1 + err2
    if total_err < best_err:
      best_err = total_err
      best_sep = sep
  return best_sep - step

def main():
  assert os.path.exists(args.input)
  last_page = None if args.n_pages is None else args.first_page + args.n_pages - 1

  st = time.monotonic()
  imgs = convert_from_path(args.input, thread_count=8, first_page=args.first_page,
                           last_page=last_page, grayscale=True, dpi=DPI)
  print(f"[INFO] convert to images done. time cost: {time.monotonic() - st:.4f}s")
  print(f"[INFO] {len(imgs)} pages in total. start from page {args.first_page}")
  arrs = [np.array(img) for img in imgs]
  print(f"[INFO] page size: {arrs[0].shape}")

  st = time.monotonic()
  if WORKERS == 1 or len(arrs) == 1:
    newarrs = [process(arr) for arr in arrs]
  elif WORKERS > 1:
    newarrs = multiprocs(process, arrs, n_workers=WORKERS, progress=True)
  scores = [evaluate(arr, newarr, i+args.first_page) for i, (arr, newarr) in enumerate(zip(arrs, newarrs))]
  print(f"[INFO] mean score: {100*np.mean(scores):.2f}%")
  print(f"[INFO] time cost: {time.monotonic() - st:.4f}s")

  # dump to pdf file
  layout_fun = img2pdf.get_fixed_dpi_layout_fun((DPI, DPI))
  with tempfile.TemporaryDirectory() as td:
    paths = [os.path.join(td, f"{i}.tiff") for i in range(len(newarrs))]
    for path, arr in zip(paths, newarrs):
      Image.fromarray(arr).save(path)
    with open(args.output, "wb") as f:
      f.write(img2pdf.convert(paths, layout_fun=layout_fun))
  print(f"[INFO] save to {args.output}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=str, required=True)
  parser.add_argument("--output", type=str, required=True)
  parser.add_argument("--first_page", type=int, default=1)
  parser.add_argument("--n_pages", type=int, default=None)
  args = parser.parse_args()
  main()
