from model.tool.regress_target import reverse_regress_target_tf
from data.pascal_voc import pascal_voc
from model.tool.timer import Timer
from model.GiraffeDet import GiraffeDet, DetectHead
from model.Loss import Loss
from evalue import voc_eval
import config as cfg 

import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import os
import sys
ckecpoint_file = cfg.ckecpoint_file
restore_path = cfg.val_restore_path
max_epoch = cfg.max_epoch

val_data = pascal_voc('test_', False, cfg.test_img_path, cfg.test_label_path, cfg.test_img_txt, True)


input_ = tf.placeholder(tf.float32, shape = [1, cfg.image_size, cfg.image_size, 3])
get_boxes = tf.placeholder(tf.float32, shape = [1, 80, 5])
num_boxes = tf.placeholder(tf.int32, shape = [1,])
imsize = tf.placeholder(tf.float32, shape = [None,])
out, all_anchors = GiraffeDet(base_anchor=cfg.base_anchor,
 scale=cfg.scale, aspect_ratio=cfg.aspect_ratio, class_num=len(cfg.classes)-1, is_training=True).forward(input_)

total_loss, regularization_loss, loc_loss, cls_loss, normalizer, matches = Loss(get_boxes, num_boxes,
 out, all_anchors, imsize)._loss() 

with tf.name_scope('detection'): 
    _cls_scores, _cls_classes, _boxes = DetectHead(cfg.score_threshold, cfg.nms_iou_threshold,
        cfg.max_detection_boxes_num, ).forward(inputs_0=tf.expand_dims(out[0], axis=0), inputs_1=tf.expand_dims(out[1], axis=0), 
        anchor=all_anchors, 
        imgsize=cfg.image_size, 
        class_num=len(cfg.classes)-1,
        score_thresh=cfg.score_threshold,
        nms_thresh=cfg.nms_iou_threshold)
    boxes, scores, labels = _boxes, _cls_scores, _cls_classes


if not os.path.exists(ckecpoint_file):
    os.makedirs(ckecpoint_file)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_writer = tf.summary.FileWriter(ckecpoint_file, sess.graph)

if restore_path is not None:
    print('Restoring weights from: ' + restore_path)
    restorer = tf.train.Saver()
    restorer.restore(sess, restore_path)
    
if __name__ == '__main__':

    val_timer = Timer()

    epoch = int(restore_path.split('-')[-1])//1000
    print('epoch', epoch)
    val_pred = []
    gt_dict = {}
    val_rloss = 0
    val_closs = 0
    val_clone_loss = 0
    for val_step in range(1, cfg.test_num+1):
        val_timer.tic()
        val_images, val_labels,val_imnm, val_num_boxes, imsize_ = val_data.get()
        val_feed_dict = {input_: val_images,
                         get_boxes: val_labels,
                         num_boxes: val_num_boxes,
                         imsize:imsize_
                        }

        b, s, l, valrloss_, valcloss_, valtotal_loss = sess.run([boxes, scores, labels, loc_loss, cls_loss, total_loss],
                                                      feed_dict = val_feed_dict)
        
        val_rloss += valrloss_/cfg.test_num
        val_closs += valcloss_/cfg.test_num
        val_clone_loss += (valrloss_ + valcloss_)/cfg.test_num

        for i in range(cfg.batch_size):
            pred_b = b[i]
            pred_s = s[i]
            pred_l = l[i]
            for j in range(pred_b.shape[0]):
                if pred_l[j] >=0 :
                    val_pred.append([val_imnm[i], pred_b[j][0], pred_b[j][1], pred_b[j][2], pred_b[j][3], pred_s[j], pred_l[j]+1])
            single_gt_num = np.where(val_labels[i][:,0]>0)[0].shape[0]
            box = np.hstack((val_labels[i][:single_gt_num, 1:], np.reshape(val_labels[i][:single_gt_num, 0], (-1,1)))).tolist()
            gt_dict[val_imnm[i]] = box 

        val_timer.toc()    
        sys.stdout.write('\r>> ' + 'val_nums '+str(val_step)+str('/')+str(cfg.test_num+1))
        sys.stdout.flush()

    print('curent val speed: ', val_timer.average_time, 'val remain time: ', val_timer.remain(val_step, cfg.test_num+1))
    print('val mean regress loss: ', val_rloss, 'val mean class loss: ', val_closs, 'val mean total loss: ', val_clone_loss)
    mean_rec = 0
    mean_prec = 0
    mAP = 0

    for classidx in range(1, cfg.class_num):#从1到20，对应[bg,...]21个类（除bg）
        rec, prec, ap = voc_eval(gt_dict, val_pred, classidx, iou_thres=0.5, use_07_metric=False)
        print(cfg.classes[classidx] + ' ap: ', ap)
        mean_rec += rec[-1]/(cfg.class_num-1)
        mean_prec += prec[-1]/(cfg.class_num-1)
        mAP += ap/(cfg.class_num-1)

    val_total_summary2 = tf.Summary(value=[
        tf.Summary.Value(tag="val/loss/class_loss", simple_value=val_closs),
        tf.Summary.Value(tag="val/loss/regress_loss", simple_value=val_rloss),
        tf.Summary.Value(tag="val/loss/clone_loss", simple_value=val_clone_loss),
        tf.Summary.Value(tag="val/mA", simple_value=mAP),
        tf.Summary.Value(tag="val/mRecall", simple_value=mean_rec),
        tf.Summary.Value(tag="val/mPrecision", simple_value=mean_prec),
     ])
    summary_writer.add_summary(val_total_summary2, epoch)
    print('Epoch: ' + str(epoch), 'mAP: ', mAP)
    print('Epoch: ' + str(epoch), 'mRecall: ', mean_rec)
    print('Epoch: ' + str(epoch), 'mPrecision: ', mean_prec)
