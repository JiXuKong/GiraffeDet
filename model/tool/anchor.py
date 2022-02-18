import numpy as np
# import tensorflow as tf
import config as cfg

scale = np.array([2**0, 2**(1/3), 2**(2/3)])
aspect_ratio = np.array([1/2, 1, 2])

def anchor_(base_size, scale, aspect_ratio, level = None):
    if level is not None:
        anchorlist = [21,14, 25,22, 41,18, 22,38, 39,24, 34,30, 50,27, 34,44, 43,36, 63,30, 46,46, 54,40, 48,57, 60,51, 83,50, 70,68, 96,85, 145,122]#[w, h]
        temp = []
        for i in range(len(anchorlist)//2):
            w = anchorlist[2*i]*2
            h = anchorlist[2*i+1]*2
            temp.append([-w/2, -h/2, w/2, h/2])
        all_anchors = [] 
        for j in range(3):
            temp1 = []
            for t in range(6):
                temp1.append(temp[6*j+t])
            all_anchors.append(temp1)
#         print(all_anchors)
        anchor = np.array(all_anchors[level])
#         print(anchor)
    else:
        area = base_size*base_size*scale**2
        w = np.sqrt(area.reshape((area.shape[0],1))/aspect_ratio.reshape(1,aspect_ratio.shape[0]))
        h = aspect_ratio*w
        w = w.transpose()
        h = h.transpose()
        w = w.reshape(-1)
        h = h.reshape(-1)
        anchor = np.vstack((-w/2, -h/2, w/2, h/2)).transpose()
        # print(anchor)
    return anchor

def generate_anchor_(base_size, scale, aspect_ration, feature_size, level = None):
    # print(feature_size)
    base_anchor = anchor_(base_size, scale, aspect_ration, level)
#     print(base_anchor)
    stride = [cfg.image_size//feature_size[0], cfg.image_size//feature_size[1]]
    grid = np.array([[j*stride[0]+stride[0]/2, i*stride[1]+stride[1]/2, j*stride[0]+stride[0]/2, i*stride[1]+stride[1]/2] for i in range(feature_size[1]) for j in range(feature_size[0])])
    generate_anchor = grid.reshape((-1, 1 ,4)) + base_anchor.reshape(1, -1, 4)
    generate_anchor = generate_anchor.reshape(-1, 4)
#     normal to (0,1)
    
    return generate_anchor



# base_anchor_ = anchor_(cfg.base_anchor[0], cfg.scale, cfg.aspect_ratio)/800
# print(base_anchor_)
# gt = np.array([[-78, -35, 44, 65]])/800
# import torch
# b_torch = torch.from_numpy(base_anchor_)
# g_torch = torch.from_numpy(gt)
# print(torch.cdist(b_torch.float(), g_torch.float(), p=1))
# import tensorflow as tf
# def iou_tf(anchors, gt):
#     '''
#     anchors:[N, 4],xmin, ymin, xmax, ymax
#     gt:[M, 4]
#     return iou: [M, N]
#     '''
#     #[N, 4]:[1, N, 4]
#     a = tf.expand_dims(anchors, axis = [0])
#     #[M, 4] [M, 1, 4]
#     g = tf.expand_dims(gt, axis = [1])
#     a = tf.cast(a,dtype=tf.float32)
#     g = tf.cast(g,dtype=tf.float32)
#     #[-1,2]
#     axymin, axymax = tf.split(a, [2,2], -1)
#     gxymin, gxymax = tf.split(g, [2,2], -1)
    
#     #[M,N,2]
#     xymin = tf.maximum(axymin, gxymin)
#     xymax = tf.minimum(axymax, gxymax)
#     #[-1,2]
#     hw = tf.maximum(xymax-xymin, 0)
#     #[N,N,1]
#     h, w = tf.split(hw, [1,1], -1)
#     #[M,N,1]
#     area =h*w
    
#     a_hw = axymax - axymin
#     a_h, a_w = tf.split(a_hw, [1,1], -1)
#     #[-1,1]
#     a_area = a_h*a_w
    
#     g_hw = gxymax - gxymin
#     g_h, g_w = tf.split(g_hw, [1,1], -1)
#     #[-1,1]
#     g_area = g_h*g_w
    
#     iou = area/(a_area + g_area - area)
#     iou = tf.squeeze(iou, -1)
# #     print(iou.get_shape().as_list())
#     return iou
# # from model.tool.IOU import iou_tf
# def tf_cdist(a, b, p):
#     '''
#     a: P×M 
#     b: R×M
#     return: P×R
#     '''
#     a = tf.expand_dims(a, axis=1)#P×1xM 
#     b = tf.expand_dims(b, axis=0)#1xRxM
#     c = tf.math.pow(tf.reduce_sum(tf.abs(a-b)**p, axis=2, keepdims=False), 1/p)
#     return c
# b_tf = tf.convert_to_tensor(base_anchor_)
# g_tf = tf.convert_to_tensor(gt)
# tf_res = tf_cdist(tf.cast(b_tf, tf.float32), tf.cast(g_tf, tf.float32), p=1)
# tf_iou = iou_tf(tf.cast(b_tf, tf.float32), tf.cast(g_tf, tf.float32))
# sess = tf.Session()
# r = sess.run(tf_res)
# r1 = sess.run(tf_iou)
# print(r, r1)