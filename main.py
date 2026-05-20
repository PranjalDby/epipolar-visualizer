# =========================
# main.py
# =========================

# type: ignore

import cv2
import numpy as np

from src.utils import fundamental_matrix_compt, get_matches
from src.utils import *
from src.ui import EpipolarUI


# ----------------------------------------
# Load images
# ----------------------------------------

img1 = cv2.imread("./assets/st1.jpg", 0)
img2 = cv2.imread("./assets/st2.png", 0)

# resize if necessary
img2 = cv2.resize(
    img2,
    (img1.shape[1], img1.shape[0])
)

# ----------------------------------------
# Feature matching
# ----------------------------------------

kp1, kp2, matches, pts1, pts2 = get_matches(
    img1,
    img2
)

print("Total matches:", len(matches))

# ----------------------------------------
# Fundamental matrix
# ----------------------------------------

F, mask = fundamental_matrix_compt(
    pts1,
    pts2
)

print("\nFundamental Matrix:\n")
print(F)

# ----------------------------------------
# Keep only inliers
# ----------------------------------------

pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

print("\nInlier matches:", len(pts1))

# ----------------------------------------
# Verify epipolar constraint
# ----------------------------------------

x1 = np.array([
    pts1[0][0],
    pts1[0][1],
    1
])

x2 = np.array([
    pts2[0][0],
    pts2[0][1],
    1
])

error = x2.T @ F @ x1

print("\nEpipolar Constraint Value:")
print(error)

# ----------------------------------------
# Launch UI
# ----------------------------------------

ui = EpipolarUI(
    img1,
    img2,
    F,
    pts1,
    pts2
)

ui.run()