import cv2
import numpy as np


def motion_blur(img, degree=10, angle=20):
    # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M,
                                        (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(img, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred


def defocus_blur(img, degree=10):
    return cv2.GaussianBlur(img, ksize=(degree, degree), sigmaX=0, sigmaY=0)


def circle_mask(size, center, radius):
    mask = np.zeros(size, dtype='uint8')
    mask = cv2.circle(mask, center, radius, 1, -1)
    return mask


if __name__ == "__main__":
    img = cv2.imread('00003_20160602171443_I_O_R_X_0_18.jpg.jpg',
                     cv2.IMREAD_GRAYSCALE)
    bimg = motion_blur(img, 20, 45)
    cv2.imshow('blur', bimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
