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


def circle_mask(size, center, radius, threshold=0):
    mask = np.zeros(size, dtype='uint8')
    mask = cv2.circle(mask, center, radius, 1, -1)
    if int(threshold) > 0:
        dist = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        mask = np.exp(-np.square(dist) / (2 * np.square(threshold))) * (dist < 3 * threshold)
    return mask.astype(np.float32)


def random_mask(size):
    num_point = np.random.randint(1, 5)
    mask = np.zeros((num_point, *size))
    for idx in range(num_point):
        center = (np.random.randint(0, size[0]), np.random.randint(0, size[1]))
        mask[idx, :, :] = cv2.circle(mask[idx, :, :], center, np.random.randint(1, 6), 1, -1)
    mask = mask.sum(axis=0)
    mask = mask / np.maximum(mask.max(), 1)
    dist = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    mask = np.exp(-np.square(dist) / (2 * np.square(50))) * (dist < 150)
    return mask


def contrast_and_brightness(img, alpha, beta):
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
    # dst = np.clip(img * alpha - beta, 0, 255)
    return dst


def imjpgcompress(img, num=10, q=60):
    for _ in range(num):
        _, encimg = cv2.imencode('.jpg', img, [cv2.IMREAD_IGNORE_ORIENTATION, q])
        img = cv2.imdecode(encimg, 0)
    return img


if __name__ == "__main__":
    img = cv2.imread('E:\\Dataset\\UniNet-test\\Image\\S2001L02.jpg',
                     cv2.IMREAD_GRAYSCALE)
    line = 'S2001L02, 0 242 349 48 248 349 96'
    line = [int(x) for x in line.split(' ')[1:]]
    center = (line[5], line[4])
    radius = line[6]
    # mask = random_mask(img.shape)
    # img = contrast_and_brightness(img, 5, 20) * mask + img * (1 - mask)
    img = imjpgcompress(img, num=10, q=5)
    cv2.imshow('img', img.astype(np.uint8))
    # mask = random_mask(img.shape)
    # cv2.imshow('img', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
