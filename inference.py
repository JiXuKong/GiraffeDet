from model.tool.regress_target import reverse_regress_target_tf
from model.tool.pred_xml import txt_2_xml
from model.RetinaNet import Retinanet
from model.tool.timer import Timer
from model.tool.NMS import gpu_nms
import config as cfg

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import copy
import cv2
import os

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque']
global_step = tf.train.get_or_create_global_step()
net = Retinanet(False, global_step)

restore_path = cfg.val_restore_path
g_list = tf.global_variables()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
    restorer = tf.train.Saver(g_list)
    restorer.restore(sess, restore_path)
    

    
if __name__ == '__main__':
    img_path = ''
    saved_img = ''
    saved_xml = ''
    
    total_timer = Timer()

    pred_classification_target_list, pred_regress_target_list = net.pred_classification_target_list, net.pred_regress_target_list
    
    pred_classification_target_list = tf.reshape(pred_classification_target_list, [-1, cfg.class_num-1])
    pred_classification_target_list = tf.nn.sigmoid(pred_classification_target_list)
    pred_regress_target_list = tf.reshape(pred_regress_target_list, [-1, 4])

    pred_regress_target_list = reverse_regress_target_tf(pred_regress_target_list, net.anchor, [cfg.image_size, cfg.image_size])
    pred_regress_i = pred_regress_target_list
    pred_classification_i = pred_classification_target_list
    nms_box, nms_score, nms_label = gpu_nms(pred_regress_i, pred_classification_i, cfg.class_num-1, 100, 0.4, 0.45)
    
    with open(cfg.train_img_txt, 'r') as f:
        image_index = [x.strip() for x in f.readlines()]
    
    for fil in os.listdir(img_path):
        if fil.split('.jpg')[0] in image_index:
            print("Pass ", fil)
            continue
        imgnm = os.path.join(img_path, fil)

        img = cv2.imread(imgnm)

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
        feed_dict = {
            net.image: img
                    }


        b, s, l = sess.run([nms_box, nms_score, nms_label], feed_dict = feed_dict)
        pred_b = b
        pred_s = s
        pred_l = l

        saved_box = []
        for j in range(pred_b.shape[0]):
            if (pred_l[j]>=0):
                
                x1,y1, x2, y2 = pred_b[j][0]*resize_scale_x, pred_b[j][1]*resize_scale_y, pred_b[j][2]*resize_scale_x, pred_b[j][3]*resize_scale_y
                saved_box.append([cfg.classes[pred_l[j]+1], int(x1),int(y1), int(x2),int(y2)])
                cv2.rectangle(img_orig,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
                cv2.putText(img_orig, cfg.classes[pred_l[j]+1] + str(pred_s[j])[:4],(int(x1),int(y1)),cv2.FONT_HERSHEY_PLAIN,2.0,(255,255,255),2,1)
        cv2.imwrite(os.path.join(saved_img, fil), img_orig)

        imageList = {
                    'pose':'Unspecified',
                    'truncated':0,
                    'difficult':0,
                    'img_w':int(x2)-int(x1),
                    'img_h':int(y2)-int(y1),
                    'image_path':imgnm,
                    'img_name':fil.split('.')[0],
                    'object':saved_box
                            }
        txt_2_xml(imageList, saved_xml, fil.split('.')[0])