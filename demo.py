from model.tool.draw_box_in_img import STANDARD_COLORS
from model.GiraffeDet import GiraffeDet, DetectHead
from model.tool.timer import Timer
import config as cfg

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy
import cv2

input_ = tf.placeholder(tf.float32, shape = [1, cfg.image_size, cfg.image_size, 3])

out, all_anchors = GiraffeDet(base_anchor=cfg.base_anchor,
 scale=cfg.scale, aspect_ratio=cfg.aspect_ratio, class_num=len(cfg.classes)-1, is_training=True).forward(input_)

with tf.name_scope('detection'): 
    # print('out', out)
    _cls_scores, _cls_classes, _boxes = DetectHead(0.3, cfg.nms_iou_threshold,
        cfg.max_detection_boxes_num, ).forward(inputs_0=tf.expand_dims(out[0], axis=0), inputs_1=tf.expand_dims(out[1], axis=0), 
        anchor=all_anchors, 
        imgsize=cfg.image_size, 
        class_num=len(cfg.classes)-1,
        score_thresh=0.3,
        nms_thresh=0.3)
    nms_box, nms_score, nms_label = _boxes[0], _cls_scores[0], _cls_classes[0]#for one img

restore_path = cfg.val_restore_path
g_list = tf.global_variables()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
    restorer = tf.train.Saver(g_list)
    restorer.restore(sess, restore_path)
    

    
if __name__ == '__main__':
    img_path = './asset/duche3.jpeg'
    saved_img ='./asset/'+'pred' + img_path.split('/')[-1]
    print(saved_img)
    
    # total_timer = Timer()

    img = cv2.imread(img_path)

    y, x = img.shape[0:2]

    resize_scale_x = x/cfg.image_size
    resize_scale_y = y/cfg.image_size
    img_orig = copy.deepcopy(img)

    img = cv2.resize(img,(cfg.image_size,cfg.image_size))
    img=img[:,:,::-1]
    img=img.astype(np.float32, copy=False)
    mean = np.array([123.68, 116.779, 103.979])
    mean = mean.reshape(1,1,3)
    img = img - mean
    img = np.reshape(img, (1, cfg.image_size, cfg.image_size, 3))
    feed_dict = {input_: img
                        }
    b, s, l = sess.run([nms_box, nms_score, nms_label], feed_dict = feed_dict)
    print(b, s, l)
    pred_b = b
    pred_s = s
    pred_l = l
    plt.figure(figsize=(20,20))
    plt.imshow(np.asarray(img_orig[:,:,::-1], np.uint8))
    plt.axis('off') 
    current_axis = plt.gca()


    for j in range(pred_b.shape[0]):
        if (pred_s[j]>=0.1):
            print(pred_l[j], pred_s[j])
            x1,y1, x2, y2 = pred_b[j][0]*resize_scale_x, pred_b[j][1]*resize_scale_y, pred_b[j][2]*resize_scale_x, pred_b[j][3]*resize_scale_y
            cls_ = pred_l[j]+1
            cls_name = str(cfg.classes[pred_l[j]+1])
            color = STANDARD_COLORS[cls_]
            current_axis.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, color=color, fill=False, linewidth=2))
            current_axis.text(x1, y1, cls_name + str(pred_s[j])[:5], size='x-large', color='white', bbox={'facecolor':'green', 'alpha':0.5})
    plt.savefig(saved_img)
    plt.show()
