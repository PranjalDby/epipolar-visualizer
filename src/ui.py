# =========================
# src/ui.py
# =========================

# type: ignore

import cv2
import numpy as np

from src.utils import draw_epiline


class EpipolarUI:

    def __init__(
        self,
        img1,
        img2,
        F,
        pts1=None,
        pts2=None
    ):

        self.img1 = img1
        self.img2 = img2
        self.F = F

        self.pts1 = pts1
        self.pts2 = pts2

    # ----------------------------------------
    # Find nearest matched point
    # ----------------------------------------

    def find_nearest_match_left(self, x, y):

        if self.pts1 is None:
            return None

        click_point = np.array([x, y])

        distances = np.linalg.norm(
            self.pts1 - click_point,
            axis=1
        )

        idx = np.argmin(distances)

        return (
            self.pts1[idx],
            self.pts2[idx]
        )

    def find_nearest_match_right(self, x, y):

        if self.pts2 is None:
            return None

        click_point = np.array([x, y])

        distances = np.linalg.norm(
            self.pts2 - click_point,
            axis=1
        )

        idx = np.argmin(distances)

        return (
            self.pts1[idx],
            self.pts2[idx]
        )

    # ----------------------------------------
    # LEFT IMAGE CLICK
    # x -> l' = Fx
    # ----------------------------------------

    def click_left(self, event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        point = np.array(
            [x, y, 1],
            dtype=np.float32
        )

        # epiline in right image
        line = self.F @ point

        # normalize
        norm = np.sqrt(
            line[0]**2 + line[1]**2
        )

        if norm > 1e-8:
            line = line / norm

        result = draw_epiline(
            self.img2,
            line
        )

        # nearest feature correspondence
        nearest = self.find_nearest_match_left(x, y)

        if nearest is not None:

            pt1, pt2 = nearest

            # draw corresponding point in right image
            cv2.circle(
                result,
                (int(pt2[0]), int(pt2[1])),
                6,
                (255, 0, 0),
                -1
            )

            # draw clicked point in left image
            left_vis = cv2.cvtColor(
                self.img1.copy(),
                cv2.COLOR_GRAY2BGR
            )

            cv2.circle(
                left_vis,
                (int(pt1[0]), int(pt1[1])),
                6,
                (0, 0, 255),
                -1
            )

            cv2.imshow(
                "Image 1",
                left_vis
            )

        cv2.imshow(
            "Epipolar Line Right",
            result
        )

    # ----------------------------------------
    # RIGHT IMAGE CLICK
    # x' -> l = F^T x'
    # ----------------------------------------

    def click_right(self, event, x, y, flags, param):

        if event != cv2.EVENT_LBUTTONDOWN:
            return

        point = np.array(
            [x, y, 1],
            dtype=np.float32
        )

        # reverse direction
        line = self.F.T @ point

        # normalize
        norm = np.sqrt(
            line[0]**2 + line[1]**2
        )

        if norm > 1e-8:
            line = line / norm

        result = draw_epiline(
            self.img1,
            line
        )

        # nearest feature correspondence
        nearest = self.find_nearest_match_right(x, y)

        if nearest is not None:

            pt1, pt2 = nearest

            # draw corresponding point in left image
            cv2.circle(
                result,
                (int(pt1[0]), int(pt1[1])),
                6,
                (255, 0, 0),
                -1
            )

            # draw clicked point in right image
            right_vis = cv2.cvtColor(
                self.img2.copy(),
                cv2.COLOR_GRAY2BGR
            )

            cv2.circle(
                right_vis,
                (int(pt2[0]), int(pt2[1])),
                6,
                (0, 0, 255),
                -1
            )

            cv2.imshow(
                "Image 2",
                right_vis
            )

        cv2.imshow(
            "Epipolar Line Left",
            result
        )

    # ----------------------------------------
    # RUN UI
    # ----------------------------------------

    def run(self):

        cv2.imshow(
            "Image 1",
            self.img1
        )

        cv2.imshow(
            "Image 2",
            self.img2
        )

        cv2.setMouseCallback(
            "Image 1",
            self.click_left
        )

        cv2.setMouseCallback(
            "Image 2",
            self.click_right
        )

        print("\n")
        print("Click on either image")
        print("Epipolar line will appear in opposite image")
        print("\n")

        cv2.waitKey(0)

        cv2.destroyAllWindows()