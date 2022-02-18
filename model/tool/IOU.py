import numpy as np
import tensorflow as tf
def iou(anchors, gt):
    
    '''
    anchors:[N, 4],xmin, ymin, xmax, ymax
    gt:[batch, M, 4]
    return iou: [batch, M, N]
    '''
    #[N, 4]:[1, 1, N, 4]
    a = np.expand_dims(anchors, axis = [0,1])

    g = np.expand_dims(gt, axis = 2)

    xymin = np.maximum(a[...,:2], g[...,:2])
    xymax = np.minimum(a[...,2:], g[...,2:])
    hw = np.maximum(xymax-xymin, 0)

    area =hw[..., 0]*hw[..., 1]

    a_hw = a[..., 2:] - a[..., :2]
    a_area = a_hw[..., 0]*a_hw[..., 1]
    g_hw = g[..., 2:] - g[..., :2]
    g_area = g_hw[..., 0]*g_hw[..., 1]
    iou = area/(a_area + g_area - area)
    return iou

def iou_tf(anchors, gt):
    '''
    anchors:[N, 4],xmin, ymin, xmax, ymax
    gt:[M, 4]
    return iou: [M, N]
    '''
    #[N, 4]:[1, N, 4]
    a = tf.expand_dims(anchors, axis = [0])
    #[M, 4] [M, 1, 4]
    g = tf.expand_dims(gt, axis = [1])
    a = tf.cast(a,dtype=tf.float32)
    g = tf.cast(g,dtype=tf.float32)
    #[-1,2]
    axymin, axymax = tf.split(a, [2,2], -1)
    gxymin, gxymax = tf.split(g, [2,2], -1)
    
    #[M,N,2]
    xymin = tf.maximum(axymin, gxymin)
    xymax = tf.minimum(axymax, gxymax)
    #[-1,2]
    hw = tf.maximum(xymax-xymin, 0)
    #[N,N,1]
    h, w = tf.split(hw, [1,1], -1)
    #[M,N,1]
    area =h*w
    
    a_hw = axymax - axymin
    a_h, a_w = tf.split(a_hw, [1,1], -1)
    #[-1,1]
    a_area = a_h*a_w
    
    g_hw = gxymax - gxymin
    g_h, g_w = tf.split(g_hw, [1,1], -1)
    #[-1,1]
    g_area = g_h*g_w
    
    iou = area/(a_area + g_area - area)
    iou = tf.squeeze(iou, -1)
#     print(iou.get_shape().as_list())
    return iou

def iou_1by1(box1, box2):
    '''
    box1:[m, 4]
    box2:[m,4]
    return [m,]
    '''
    box1 = tf.cast(box1,dtype=tf.float32)
    box2 = tf.cast(box2,dtype=tf.float32)
    #[m,2]
    box1xymin, box1xymax = tf.split(box1, [2,2], -1)
    box2xymin, box2xymax = tf.split(box2, [2,2], -1)
    #[M,2]
    xymin = tf.maximum(box1xymin, box2xymin)
    xymax = tf.minimum(box1xymax, box2xymax)
    #[M,2]
    hw = tf.maximum(xymax-xymin, 0)
    #[N,1]
    h, w = tf.split(hw, [1,1], -1)
    #[N,1]
    area =h*w
    #[m,2]
    box1_hw = box1xymax - box1xymin
    box1_h, box1_w = tf.split(box1_hw, [1,1], -1)
    #[-1,1]
    box1_area = box1_h*box1_w
    #[m,2]
    box2_hw = box2xymax - box2xymin
    box2_h, box2_w = tf.split(box2_hw, [1,1], -1)
    #[-1,1]
    box2_area = box2_h*box2_w
    #[-1,1]
    iou = area/(box1_area + box2_area - area)
    iou = tf.squeeze(iou, -1)
    return iou

# def iou_tf(anchors, gt):
#     '''
#     anchors:[N, 4],xmin, ymin, xmax, ymax
#     gt:[M, 4]
#     return iou: [M, N]
#     '''
#     #[N, 4]:[1, N, 4]
#     a = tf.expand_dims(anchors, axis = [0])
#     a = tf.expand_dims(a, axis = [1])
#     #[batch, M, 4] [batch, M, 1, 4]
#     g = tf.expand_dims(gt, axis = 2)
    
#     axymin, axymax = tf.split(a, [2,2], -1)
#     gxymin, gxymax = tf.split(g, [2,2], -1)
    
#     xymin = tf.maximum(axymin, gxymin)
#     xymax = tf.minimum(axymax, gxymax)
    
#     hw = tf.maximum(xymax-xymin, 0)
#     h, w = tf.split(hw, [1,1], -1)
#     area =h*w

#     a_hw = axymax - axymin
#     a_h, a_w = tf.split(a_hw, [1,1], -1)
#     a_area = a_h*a_w
    
#     g_hw = gxymax - gxymin
#     g_h, g_w = tf.split(g_hw, [1,1], -1)
#     g_area = g_h*g_w
    
#     iou = area/(a_area + g_area - area)
#     iou = tf.squeeze(iou, -1)
#     return iou
