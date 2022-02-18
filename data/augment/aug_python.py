import cv2
import numpy as np
import random
import math

def matrix_iou(a,b):

    """
    return iou of a and b, numpy version for data augenmentation
    
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i)



def _crop(image, boxes, labels):
    height, width, = image.shape[:2]    
    if len(boxes.shape) == 0:
        return image, boxes, labels
    
    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))
        if mode is None:
            return image, boxes, labels
        
        
        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')
            
        for _ in range(50):
            scale = random.uniform(0.3, 1.)
            min_ratio = max(0.5, scale*scale)
            max_ratio = min(2, 1./scale/scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale*ratio*width)
            h = int((scale/ratio)*height)
            
            l = random.randrange(width - w)
            t = random.randrange(height - h)
            
            roi = np.array((l, t, l + w, t + h))
            iou = matrix_iou(boxes, roi[np.newaxis])
            
            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]) \
                .all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue


            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]
            return image_t, boxes_t, labels_t

    
def random_color_distort(img, brightness_delta=64, hue_vari=18, sat_vari=0.5, val_vari=0.5):
    '''
    randomly distort image color. Adjust brightness, hue, saturation, value.
    param:
        img: a BGR uint8 format OpenCV image. HWC format.
    '''

    def random_hue(img_hsv, hue_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            hue_delta = np.random.randint(-hue_vari, hue_vari)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0] + hue_delta) % 180
        return img_hsv

    def random_saturation(img_hsv, sat_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            sat_mult = 1 + np.random.uniform(-sat_vari, sat_vari)
            img_hsv[:, :, 1] *= sat_mult
        return img_hsv

    def random_value(img_hsv, val_vari, p=0.5):
        if np.random.uniform(0, 1) > p:
            val_mult = 1 + np.random.uniform(-val_vari, val_vari)
            img_hsv[:, :, 2] *= val_mult
        return img_hsv

    def random_brightness(img, brightness_delta, p=0.5):
        if np.random.uniform(0, 1) > p:
            img = img.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness_delta, brightness_delta))
            img = img + brightness_delta
        return np.clip(img, 0, 255)

    # brightness
    img = random_brightness(img, brightness_delta)
    img = img.astype(np.uint8)

    # color jitter
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        img_hsv = random_value(img_hsv, val_vari)
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
    else:
        img_hsv = random_saturation(img_hsv, sat_vari)
        img_hsv = random_hue(img_hsv, hue_vari)
        img_hsv = random_value(img_hsv, val_vari)

    img_hsv = np.clip(img_hsv, 0, 255)
    img = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
        Refer from: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """

    def __init__(self, n_holes, ratio, fill_value=0., p=1.0):
        self.n_holes = n_holes
        self.ratio = ratio
        assert 0. <= fill_value <= 1., "the fill value is in a range of 0 to 1"
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img, targets):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if np.random.random() <= self.p:
            h = img.size(1)
            w = img.size(2)

            h_cutout = int(self.ratio * h)
            w_cutout = int(self.ratio * w)

            for n in range(self.n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)

                y1 = np.clip(y - h_cutout // 2, 0, h)
                y2 = np.clip(y + h_cutout // 2, 0, h)
                x1 = np.clip(x - w_cutout // 2, 0, w)
                x2 = np.clip(x + w_cutout // 2, 0, w)

                img[:, y1: y2, x1: x2] = self.fill_value  # Zero out the selected area
                # Remove targets that are in the selected area
                keep_target = []
                for target_idx, target in enumerate(targets):
                    _, _, target_x, target_y, target_w, target_l, _, _ = target
                    if (x1 <= target_x * w <= x2) and (y1 <= target_y * h <= y2):
                        continue
                    keep_target.append(target_idx)
                targets = targets[keep_target]

        return img, targets
