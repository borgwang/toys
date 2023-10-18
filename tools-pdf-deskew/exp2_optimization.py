import numpy as np
import pygmo as pg
from pdf2image import convert_from_path
from PIL import ImageDraw


def plot(img, weights):
  draw = ImageDraw.Draw(img)
  y1, x1, y2, x2, y3, x3, y4, x4 = weights
  r = 4
  for i in range(0, len(weights), 2):
    h, w = weights[i], weights[i + 1]
    draw.ellipse((w - r, h - r, w + r, h + r), fill="black")
  draw.line((x1, y1, x2, y2, x4, y4, x3, y3, x1, y1), fill="black", width=2)
  img.show()


class Prob:

  def fitness(self, weights):

    def triangle_area(p1, p2, p3):

      def dist(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

      a, b, c = dist(p1, p2), dist(p1, p3), dist(p2, p3)
      s = (a + b + c) / 2
      return (s * (s - a) * (s - b) * (s - c)) ** 0.5

    h, w = arr.shape
    y1, x1, y2, x2, y3, x3, y4, x4 = map(int, weights)
    s1 = (triangle_area((x1, y1), (x2, y2), (x3, y3)) +
          triangle_area((x2, y2), (x3, y3), (x4, y4)))
    s2 = (triangle_area((x1, y1), (x2, y2), (x4, y4)) +
          triangle_area((x1, y1), (x3, y3), (x4, y4)))
    s_11 = min(s1, s2)

    yx = np.stack(np.meshgrid(range(h), range(w)), -1).reshape(-1, 2)
    y0, x0 = yx[:, 0], yx[:, 1]
    s_22 = (triangle_area((x0, y0), (x1, y1), (x2, y2)) +
            triangle_area((x0, y0), (x2, y2), (x4, y4)) +
            triangle_area((x0, y0), (x3, y3), (x4, y4)) +
            triangle_area((x0, y0), (x3, y3), (x1, y1)))
    mask = np.isclose(s_22, s_11, atol=1)
    inside_pos, outside_pos = yx[mask, :], yx[~mask, :]
    # obj1: penalize non-white pixels outside the box
    obj1 = arr[outside_pos[:, 0], outside_pos[:, 1]].mean() / 255
    # obj2: small box
    obj2 = 1 - s_11 / (h * w)
    # obj3: more non-white pixels inside the box
    obj3 = 1 - arr[inside_pos[:, 0], inside_pos[:, 1]].mean() / 255
    # obj4: more like a retangle
    obj4 = 1 - (0.5 * ((x1 - x3) ** 2 + (x2 - x4) ** 2) / (2 * (0.25 * w) ** 2) +
                0.5 * ((y1 - y2) ** 2 + (y3 - y4) ** 2) / (2 * (0.2 * h) ** 2))
    # minimize fitness
    fitness = -(2 * obj1 + obj2 + obj3 + 2 * obj4)
    print(mask.mean(), fitness, obj1, obj2, obj3, obj4)
    return (fitness,)

  def get_bounds(self):
    h, w = arr.shape
    return ([0, 0, 0, w * 0.75, 0.8 * h, 0, 0.8 * h, 0.75 * w],
            [h * 0.2, w * 0.25, h * 0.2, w, h, 0.25 * w, h, w - 1])


doc_path = "./test.pdf"
images = convert_from_path(doc_path, thread_count=10, first_page=1, last_page=2,
                           grayscale=True, dpi=200)

img = images[0]
arr = np.array(img)

prob = pg.problem(Prob())
pop = pg.population(prob, size=50)
algo = pg.algorithm(pg.sga(gen=10))
algo.set_verbosity(1)
pop = algo.evolve(pop)
uda = algo.extract(pg.sga)
res = list(pop.champion_x)
plot(img, res)
