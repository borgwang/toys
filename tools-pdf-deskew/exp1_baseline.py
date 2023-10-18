import numpy as np
from evaluate import evaluate
from pdf2image import convert_from_path

doc_path = "./test.pdf"
imgs = convert_from_path(doc_path, thread_count=10, first_page=1, last_page=None,
                         grayscale=True, dpi=200)

def simple_rotate(img):
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


scores = []
for img in imgs:
  arr = np.array(img)
  tarr = simple_rotate(img)
  scores.append(evaluate(arr, tarr))
print(f"mean score: {np.mean(scores):.4f}")
