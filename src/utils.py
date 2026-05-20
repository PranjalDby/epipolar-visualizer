# type:ignore
import cv2 as cv # type:ignore
import numpy as np # type: ignore
# import os
# import matplotlib.pyplot as plt
# 
# 
def get_matches(img1,img2):
    """Two images with same scene but different texture or alignment"""
    # orb = cv.ORB_create(3000)
    # kp1, des1 = orb.detectAndCompute(img1,None) # calculate the keypoint using FAST and uses BREIF to calculate the descriptor
    # kp2, des2 = orb.detectAndCompute(img2,None) # calculate the same
    #     # using SIFT
    sift = cv.SIFT_create()
    kp1,des1 = sift.detectAndCompute(img1,None)
    kp2,des2 = sift.detectAndCompute(img2,None)
    # brute force matcher # object detection
    # matcher = cv.BFMatcher(cv.NORM_HAMMING,crossCheck  = True)
    # matches = matcher.match(des1,des2)
    # pts1 = np.float32(
    #        [kp1[m.queryIdx].pt for m in matches]
    # )
   
    # pts2 = np.float32(
    #     [kp2[m.trainIdx].pt for m in matches]
    # )
    # using FLANN
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    
    flann = cv.FlannBasedMatcher(
        index_params,
        search_params
    )
    
    matches = flann.knnMatch(des1, des2, k=2)
    # getting only good_matches
    good_matches = []
    
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # extracting only good points
    pts1 = np.float32([
        kp1[m.queryIdx].pt
        for m in good_matches
    ])
    
    pts2 = np.float32([
        kp2[m.trainIdx].pt
        for m in good_matches
    ])
    return kp1, kp2, good_matches, pts1, pts2

def fundamental_matrix_compt(pts1,pts2):
    F, mask = cv.findFundamentalMat(
           pts1,
           pts2,
           cv.FM_RANSAC
    )
    pts1
    return F, mask

# %%
# type: ignore

def draw_epiline(img, line):

    img_color = cv.cvtColor(
        img.copy(),
        cv.COLOR_GRAY2BGR
    )

    h, w = img.shape

    a, b, c = line

    points = []

    # x = 0
    if abs(b) > 1e-8:
        y = int(-c / b)

        if 0 <= y < h:
            points.append((0, y))

    # x = w
    if abs(b) > 1e-8:
        y = int(-(c + a * w) / b)

        if 0 <= y < h:
            points.append((w, y))

    # y = 0
    if abs(a) > 1e-8:
        x = int(-c / a)

        if 0 <= x < w:
            points.append((x, 0))

    # y = h
    if abs(a) > 1e-8:
        x = int(-(c + b * h) / a)

        if 0 <= x < w:
            points.append((x, h))

    # remove duplicates
    points = list(set(points))

    if len(points) >= 2:

        cv.line(
            img_color,
            points[0],
            points[1],
            (0, 255, 0),
            2
        )

    return img_color

# p1 = "../assets/st1.jpg"
# p2 = "../assets/st2.png"
# print(os.path.pardir)
# if __name__ == "__main__":
#     print('Checking Valid Matches')
#     img1 = cv.imread(p1, 0)
#     img2 = cv.imread(p2, 0)
#     kp1,kp2,matches,pts1,pts2 = get_matches(img1,img2)

#     drw_matc = cv.drawMatches(img1,kp1,img2,kp2,matches[:50],None)
#     plt.imshow(drw_matc)
#     plt.show()