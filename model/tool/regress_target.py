import numpy as np
import tensorflow as tf

import config as cfg
def regress_target(gt_box, anchor_box):
    '''
    gt_box: [M, 4]
    x1, y1, x2, y2
    anchor_box:[M, 4]
    x1, y1, x2, y2
    '''
    gt_box.astype(np.float32)
    anchor_box.astype(np.float32)
    x1 = gt_box[..., 0]
    y1 = gt_box[..., 1]
    x2 = gt_box[..., 2]
    y2 = gt_box[..., 3]
    w = x2 - x1 +1
    h = y2 - y1 +1
    x_ctr = x1 + w/2
    y_ctr = y1 + h/2
    
    xx1 = anchor_box[..., 0]
    yy1 = anchor_box[..., 1]
    xx2 = anchor_box[..., 2]
    yy2 = anchor_box[..., 3]
    ww = xx2 - xx1
    hh = yy2 - yy1
    xx_ctr = xx1 + ww/2
    yy_ctr = yy1 + hh/2
    
    tx = (x_ctr-xx_ctr)/w
    ty = (y_ctr-yy_ctr)/h
    tw = np.log(w/ww)
    th = np.log(h/hh)
    return np.vstack((tx,ty,tw,th)).transpose()

def regress_target_tf(gt_box, anchor_box, img_size):
    '''
    gt_box: [M, 4]
    x1, y1, x2, y2
    anchor_box:[M, 4]
    x1, y1, x2, y2
    img_size: [h, w]
    '''
    # print("img_size: ", img_size[1])
    gt_box = tf.to_float(gt_box)
    anchor_box = tf.to_float(anchor_box)
    img_size = tf.to_float(img_size)
    x1, y1, x2, y2 = tf.split(gt_box, [1,1,1,1], -1)
    x1, y1, x2, y2 = x1/img_size[0], y1/img_size[0], x2/img_size[0], y2/img_size[0]
    w = x2 - x1
    h = y2 - y1
    x_ctr = x1 + w/2
    y_ctr = y1 + h/2
    
    xx1, yy1, xx2, yy2 = tf.split(anchor_box, [1,1,1,1], -1)
    xx1, yy1, xx2, yy2 = xx1/img_size[0], yy1/img_size[0], xx2/img_size[0], yy2/img_size[0]
    ww = xx2 - xx1
    hh = yy2 - yy1
    xx_ctr = xx1 + ww/2
    yy_ctr = yy1 + hh/2
    
    tx = (x_ctr-xx_ctr)/ww
    ty = (y_ctr-yy_ctr)/hh
    tw = tf.log(w/ww)
    th = tf.log(h/hh)
#     k = tf.stack((tx,ty,tw,th), axis=0)
    if not cfg.giou_loss:
        return tf.concat([tx*5.0,ty*5.0,tw*5.0,th*5.0], axis = -1)
    else:
        return tf.concat([tx,ty,tw,th], axis = -1)
#     print('k', k.get_shape().as_list())
#     return tf.transpose(tf.squeeze(tf.stack((tx,ty,tw,th), axis=0), -1), [1,0])

def reverse_regress_target_tf(pred_box, anchor_box, img_size):
    '''
    anchor_box: [-1, 4]
    pred_box:[-1, 4]
    '''
#     batch = pred_box.get_shape().as_list()[0]
    anchor_box = tf.to_float(anchor_box)
    img_size = tf.to_float(img_size)
#     anchor_box = tf.reshape(tf.tile(anchor_box, [batch, 1]), [batch, -1, 4])
    xx1, yy1, xx2, yy2 = tf.split(anchor_box, [1,1,1,1], -1)
    xx1, yy1, xx2, yy2 = xx1/img_size[0], yy1/img_size[0], xx2/img_size[0], yy2/img_size[0]
    ww = xx2 - xx1
    hh = yy2 - yy1
    xx_ctr = xx1 + ww/2.0
    yy_ctr = yy1 + hh/2.0
    
    tx, ty, tw, th = tf.split(pred_box, [1,1,1,1], -1)
    if not cfg.giou_loss:
        tx, ty, tw, th = tx/5.0, ty/5.0, tw/5.0, th/5.0
    etw, eth = tf.exp(tw), tf.exp(th)
    h, w = hh*eth, ww*etw
    x_ctr, y_ctr = tx*ww + xx_ctr, ty*hh + yy_ctr
    x1 = x_ctr - w/2.0
    y1 = y_ctr - h/2.0
    x2 = x_ctr + w/2.0
    y2 = y_ctr + h/2.0
#     return tf.concat([x1,y1,x2,y2], axis=-1)
    return tf.stop_gradient(tf.concat([x1*img_size[0],y1*img_size[0],x2*img_size[0],y2*img_size[0]], axis=-1))


# def normolize_box(box, img_size):
#     box = tf.to_float(box)
#     img_size = tf.to_float(img_size)
#     x1, y1, x2, y2 = tf.split(box, [1,1,1,1], -1)
#     x1, y1, x2, y2 = x1/img_size[0], y1/img_size[0], x2/img_size[0], y2/img_size[0]

#     delta = tf.concat([x1,y1,x2,y2], axis=-1) 
#     return delta

# def reverse_normolize_box(delta, img_size):
#     delta = tf.to_float(delta)
#     img_size = tf.to_float(img_size)
#     x1, y1, x2, y2 = tf.split(delta, [1,1,1,1], -1)
#     # x1, y1, x2, y2 = x1/img_size[0], y1/img_size[0], x2/img_size[0], y2/img_size[0]

#     box = tf.concat([x1*img_size[0],y1*img_size[0],x2*img_size[0],y2*img_size[0]], axis=-1) 
#     return box
