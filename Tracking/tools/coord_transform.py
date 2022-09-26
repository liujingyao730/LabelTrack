import numpy as np
import cv2


def get_img2grnd(img_pxl: np.ndarray, grnd_coord: np.ndarray):
    """given the image pixel and the related ground location, 
    calculate the transform matrix.

    Args:
        img_pxl (np.ndarray): the image pixel locations.
        grnd_coord (np.ndarray): the related ground coordinate location.

    Returns:
        img2grnd: the transform matrix of 2d images. (3x3)
    """

    img2grnd, _ = cv2.findHomography(img_pxl, grnd_coord, cv2.RHO)

    return img2grnd


def img2grnd(img_pxls: np.ndarray, img2grnd: np.ndarray):
    """given the image pixels and the transform matrix, get the ground coordinate.

    Args:
        img_pxls (np.ndarray): homography coordinates of image pixel (u, v, 1). (3xn)
        img2grnd (np.ndarray): transform matrix from image pixel to coordinate. (3x3)

    Returns:
        grnd_coord (np.ndarray): homography coordinates of ground location (x, y, 1). (3xn)
    """

    grnd_coord = img_pxls @ img2grnd

    return grnd_coord 