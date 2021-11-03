import argparse

import cv2
import numpy as np
from pdf2image import convert_from_path
from PIL import Image, ImageDraw
from sklearn.linear_model import LinearRegression


class ImageRectifier:

    def __init__(self,
                 degree_lb=-1,
                 degree_ub=1 + 1e-8,
                 degree_step=0.05,
                 scale_factor=4,
                 eval_flag=True,
                 plot_flag=False):
        self.degree_lb = degree_lb
        self.degree_ub = degree_ub
        self.degree_step = degree_step

        self.scale_factor = scale_factor
        self.eval_flag = eval_flag
        self.plot_flag = plot_flag

    def run(self, arr):
        # step1: get edges and corner points
        sarr = arr[::self.scale_factor, ::self.scale_factor]
        edges = self._get_edges(sarr)
        points = {
            "tl": self._get_intersection(edges["top"], edges["left"]),
            "tr": self._get_intersection(edges["top"], edges["right"]),
            "br": self._get_intersection(edges["bottom"], edges["right"]),
            "bl": self._get_intersection(edges["bottom"], edges["left"])
        }

        # unscaled
        for name in points:
            points[name] *= self.scale_factor
        for name in edges:
            edges[name] *= self.scale_factor

        if self.eval_flag:
            self.evaluate(arr, points)

        # step2: deskew
        tarr = self._transform(arr, points)

        if self.plot_flag:
            # origin
            img = Image.fromarray(arr)
            draw = ImageDraw.Draw(img)
            for edge in edges.values():
                draw.line(tuple(edge), fill="black", width=2)
            r = 4
            for w, h in points.values():
                draw.ellipse((w - r, h - r, w + r, h + r), fill="black")
            img.show()
            # output
            Image.fromarray(tarr).show()
        return tarr

    def _transform(self, arr, points):
        oh, ow = arr.shape
        h1 = self._distance(points["tl"], points["bl"])
        h2 = self._distance(points["tr"], points["br"])
        th = max(int(h1), int(h2))
        w1 = self._distance(points["tl"], points["tr"])
        w2 = self._distance(points["bl"], points["br"])
        tw = max(int(w1), int(w2))

        dst = np.array([[0, 0], [tw - 1, 0], [tw - 1, th - 1], [0, th - 1]], dtype="float32")
        src = np.array(list(points.values()), dtype="float32")
        M = cv2.getPerspectiveTransform(src, dst)
        tarr = cv2.warpPerspective(arr, M, (tw, th))
        # padding
        pad_h, pad_w = (oh - th) // 2, (ow - tw) // 2
        tarr = np.pad(tarr, ((pad_h, oh - th - pad_h), (pad_w, ow - tw - pad_w)), constant_values=255)
        return tarr

    def _get_edges(self, arr):
        h, w = arr.shape
        edges = {}

        for side in ("top", "bottom"):
            best_var = -1
            best_line = None
            x_ = np.arange(w).astype(int)
            for degree in np.arange(self.degree_lb, self.degree_ub, self.degree_step):
                tan = np.tan(np.abs(degree) / 180 * np.pi)
                if side == "top":
                    if degree > 0:
                        y_ = tan * (w - x_)
                    else:
                        y_ = tan * x_
                else:
                    if degree > 0:
                        y_ = h - tan * x_
                    else:
                        y_ = h - tan * (w - x_)
                y_ = y_.astype(int)

                concentration = []
                offsets = np.arange(h // 6) if side == "top" else np.arange(-1, -h // 6, -1)
                for offset in offsets:
                    concentration.append(np.mean(arr[y_ + offset, x_]))

                concentration = np.array(concentration)
                variance = ((concentration[1:] - concentration[:-1]) ** 2).sum()
                #variance = np.var(concentration)
                if variance > best_var:
                    best_var = variance
                    best_offset = offsets[self._change_point_detection(concentration)]
                    best_line = ((y_[0] + best_offset, x_[0]), (y_[-1] + best_offset, x_[-1]))
            (y1, x1), (y2, x2) = best_line
            edges[side] = np.array([x1, y1, x2, y2])

        for side in ("left", "right"):
            best_var = -1
            best_line = None
            y_ = np.arange(h).astype(int)
            for degree in np.arange(self.degree_lb, self.degree_ub, self.degree_step):
                tan = np.tan(np.abs(degree) / 180 * np.pi)
                if side == "left":
                    if degree > 0:
                        x_ = (tan * y_).astype(int)
                    else:
                        x_ = (tan * (h - y_)).astype(int)
                else:
                    if degree > 0:
                        x_ = (w - tan * (h - y_)).astype(int)
                    else:
                        x_ = (w - tan * y_).astype(int)

                concentration = []
                offsets = np.arange(w // 4) if side == "left" else np.arange(-1, -w // 4, -1)
                for offset in offsets:
                    concentration.append(np.mean(arr[y_, x_ + offset]))

                variance = np.var(concentration)
                if variance > best_var:
                    best_var = variance
                    best_offset = offsets[self._change_point_detection(concentration)]
                    best_line = ((y_[0], x_[0] + best_offset), (y_[-1], x_[-1] + best_offset))
            (y1, x1), (y2, x2) = best_line
            edges[side] = np.array([x1, y1, x2, y2])
        return edges

    @staticmethod
    def _change_point_detection(points, step=1):
        points = np.array(points)[:, np.newaxis]
        xs = np.arange(len(points))[:, np.newaxis]
        best_err, best_sep = float("inf"), None
        for sep in range(step, len(points), step):
            model1 = LinearRegression()
            model1.fit(xs[:sep], points[:sep])
            err1 = (np.abs(model1.predict(xs[:sep]) - points[:sep]) ** 0.1).sum()
            model1.fit(xs[sep:], points[sep:])
            err2 = (np.abs(model1.predict(xs[sep:]) - points[sep:]) ** 0.1).sum()
            total_err = err1 + err2
            if total_err < best_err:
                best_err = total_err
                best_sep = sep
        return best_sep - step

    @staticmethod
    def _get_intersection(line1, line2):

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

    @staticmethod
    def _distance(p1, p2):
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _triangle_area(self, p1, p2, p3):
        a, b, c = self._distance(p1, p2), self._distance(p1, p3), self._distance(p2, p3)
        s = (a + b + c) / 2
        return (s * (s - a) * (s - b) * (s - c)) ** 0.5

    def evaluate(self, arr, points):
        h, w = arr.shape
        (x1, y1), (x2, y2), (x4, y4), (x3, y3) = points.values()

        s1 = (self._triangle_area((x1, y1), (x2, y2), (x3, y3)) +
              self._triangle_area((x2, y2), (x3, y3), (x4, y4)))
        s2 = (self._triangle_area((x1, y1), (x2, y2), (x4, y4)) +
              self._triangle_area((x1, y1), (x3, y3), (x4, y4)))
        s_11 = min(s1, s2)

        yx = np.stack(np.meshgrid(range(h), range(w)), -1).reshape(-1, 2)
        y0, x0 = yx[:, 0], yx[:, 1]
        s_22 = (self._triangle_area((x0, y0), (x1, y1), (x2, y2)) +
                self._triangle_area((x0, y0), (x2, y2), (x4, y4)) +
                self._triangle_area((x0, y0), (x3, y3), (x4, y4)) +
                self._triangle_area((x0, y0), (x3, y3), (x1, y1)))
        mask = np.isclose(s_22, s_11, atol=0.1)
        inside_pos, outside_pos = yx[mask, :], yx[~mask, :]
        # non-white pixels outside the box
        score1 = arr[outside_pos[:, 0], outside_pos[:, 1]].mean() / 255 - 0.9
        print(f"score1: {score1:.4f}")
        # non-white pixels inside the box
        score2 = 1 - arr[inside_pos[:, 0], inside_pos[:, 1]].mean() / 255 + 0.9
        print(f"score2: {score2:.4f}")
        score = 10 * score1 + score2
        print(f"final score: {score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=str, default="./test.pdf")
    parser.add_argument("--first_page", type=int, default=0)
    parser.add_argument("--n_pages", type=int, default=1)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--eval", type=int, default=1)
    parser.add_argument("--plot", type=int, default=1)
    args = parser.parse_args()

    imgs = convert_from_path(args.pdf, thread_count=10, first_page=args.first_page,
                             last_page=args.first_page + args.n_pages,
                             grayscale=True, dpi=args.dpi)
    rectifier = ImageRectifier(scale_factor=args.scale, plot_flag=args.plot)
    for img in imgs:
        arr = np.array(img)
        rectifier.run(arr)
