import tensorflow as tf
import numpy as np

def localization_loss(predictions, targets, weights):
    """A usual L1 smooth loss.

    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, 4],
            representing the (encoded) predicted locations of objects.
        targets: a float tensor with shape [batch_size, num_anchors, 4],
            representing the regression targets.
        weights: a float tensor with shape [batch_size, num_anchors].
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0)
    loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    return weights * tf.reduce_sum(loss, axis=2)

def balanced_l1_loss(predictions, targets, weights, beta=1.0, alpha=0.5, gamma=1.5,):
    '''
    if diff<beta: alpha / b *(b * diff + 1) * log(b * diff / beta + 1) - alpha * diff 
    if diff>beta: gamma * diff + gamma / b - alpha * beta 
    '''
    
    b = np.e**(gamma / alpha) - 1
    abs_diff = tf.abs(predictions - targets)
    abs_diff_lt_1 = tf.less(abs_diff, 1.0)
    loss = tf.where(abs_diff_lt_1, alpha / b *(b * abs_diff + 1) * tf.log(b * abs_diff / beta + 1) - alpha * abs_diff, 
                    gamma * abs_diff + gamma / b - alpha * beta)
    return weights * tf.reduce_sum(loss, axis=2)

def G_IOU_loss(predictions, targets, weights):
    '''
    predictions:batch, -1, 4
    targets:batch, -1, 4
    # Copyright (C) 2019 * Ltd. All rights reserved.
    # author : SangHyeon Jo <josanghyeokn@gmail.com>

    GIoU = IoU - (C - (A U B))/C
    GIoU_Loss = 1 - GIoU
    '''
    def GIoU(bboxes_1, bboxes_2):
        # 1. calulate intersection over union
        area_1 = (bboxes_1[..., 2] - bboxes_1[..., 0]) * (bboxes_1[..., 3] - bboxes_1[..., 1])#batch, -1
        area_2 = (bboxes_2[..., 2] - bboxes_2[..., 0]) * (bboxes_2[..., 3] - bboxes_2[..., 1])#batch, -1

        intersection_wh = tf.minimum(bboxes_1[:, :, 2:], bboxes_2[:, :, 2:]) - tf.maximum(bboxes_1[:, :, :2], bboxes_2[:, :, :2])#batch, -1, 2
        intersection_wh = tf.maximum(intersection_wh, 0)#batch, -1, 2

        intersection = intersection_wh[..., 0] * intersection_wh[..., 1]#batch, -1
        union = (area_1 + area_2) - intersection#batch, -1

        ious = intersection / tf.maximum(union, 1e-10)#batch, -1

        # 2. (C - (A U B))/C
        C_wh = tf.maximum(bboxes_1[..., 2:], bboxes_2[..., 2:]) - tf.minimum(bboxes_1[..., :2], bboxes_2[..., :2])#batch, -1, 2
        C_wh = tf.maximum(C_wh, 0.0)#batch, -1, 2
        C = C_wh[..., 0] * C_wh[..., 1]#batch, -1

        giou = ious - (C - union) / tf.maximum(C, 1e-10)
        return 1-giou
    return GIoU(predictions, targets)*weights




def focal_loss(predictions, targets, weights, gamma=2.0, alpha=0.25):
    """
    Arguments:
        predictions: a float tensor with shape [batch_size, num_anchors, num_classes],
            representing the predicted logits for each class.
        targets: a float tensor with shape [batch_size, num_anchors, num_classes],
            representing one-hot encoded classification targets.
        weights: a float tensor with shape [batch_size, num_anchors].
        gamma, alpha: float numbers.
    Returns:
        a float tensor with shape [batch_size, num_anchors].
    """
    positive_label_mask = tf.equal(targets, 1.0)

#     delta = 0.01
#     targets = (1 - delta) * targets + + delta * 1. / 2
    negative_log_p_t = tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=predictions)
#     negative_log_p_t = tf.losses.sigmoid_cross_entropy(multi_class_labels=targets, logits = predictions, label_smoothing=0)
    
    
    probabilities = tf.sigmoid(predictions)
    p_t = tf.where(positive_label_mask, probabilities, 1.0 - probabilities)
    # they all have shape [batch_size, num_anchors, num_classes]

    modulating_factor = tf.pow(1.0 - p_t, gamma)
    weighted_loss = tf.where(
        positive_label_mask,
        alpha * negative_log_p_t,
        (1.0 - alpha) * negative_log_p_t
    )
    focal_loss = modulating_factor * weighted_loss
    # they all have shape [batch_size, num_anchors, num_classes]

    return weights * tf.reduce_sum(focal_loss, axis=2)

