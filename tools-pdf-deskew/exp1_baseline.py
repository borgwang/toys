import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageFilter, ImageDraw

doc_path = "./test.pdf"
images = convert_from_path(doc_path, thread_count=10, first_page=1, last_page=2,
                           grayscale=True, dpi=200)
img = images[0]


def simple_rotate():
    w, h = img.size
    max_std = 0
    best_degree = None
    for degree in np.arange(-1.5, 1.5 + 1e-4, 0.1):
        img2 = img.rotate(degree, expand=True)
        offset_x, offset_y = w // 8, h // 8
        img3a = np.array(img2)[offset_y:-offset_y, offset_x:-offset_x]
        std = np.std(img3a.mean(axis=1))
        if std > max_std:
            max_std = std
            best_degree = degree
    tarr = np.array(img.rotate(best_degree))
    Image.fromarray(tarr).show()

simple_rotate()

