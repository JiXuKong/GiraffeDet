import copy
import numpy as np
import tensorflow as tf

import config as cfg
from model.tool.IOU import iou_tf, iou_1by1
from model.tool.regress_target import regress_target_tf

def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = tf.split(x, [1,1,1,1], -1)
    w = x1 - x0
    h = y1 - y0
    ctr_x = x0 + w/2
    ctr_y = y0 + h/2
    return tf.concat([ctr_x, ctr_y, w, h], axis = -1)

def tf_cdist(a, b, p):
    '''
    a: P×M 
    b: R×M
    return: P×R
    '''
    a = tf.cast(a, tf.float32)
    b= tf.cast(b, tf.float32)
    a = tf.expand_dims(a, axis=1)#P×1xM 
    b = tf.expand_dims(b, axis=0)#1xRxM
    c = tf.math.pow(tf.reduce_sum(tf.abs(a-b)**p, axis=2, keepdims=False), 1/p)
    return c


class OneImgUniformMatcher(object):
    '''
    Uniform Matching between the anchors and gt boxes, which can achieve
    balance in positive anchors.

    Args:
        match_times(int): Number of positive anchors for each gt box.

    '''
    def __init__(self, match_times,neg_ignor_th=0.7, pos_ignor_th=0.15):
        self.match_times = match_times
        self.pred_match_times = match_times
        self.neg_ignor_th = neg_ignor_th
        self.pos_ignor_th = pos_ignor_th
    def forward(self, anchors, targets, pred_box, img_size):
        '''
        anchors:[-1,4]
        targets[0]: box  [num, 4]
        targets[1]: cls  [num]
        pred_box:[-1, 4]
        img_size:[2]
        '''

        tgt_bbox =targets[0]#[t, 4]
        tgt_bbox = tf.reshape(tgt_bbox, [-1,4])
        gt_labels = tf.reshape(targets[1], [-1])
        anchors = tf.reshape(anchors, [-1,4])#[num_anchors, 4]
        # num_queries = tf.shape(anchors)[0]
        pred_box = tf.reshape(pred_box, [-1,4])

        cost_bbox_anchors = tf_cdist(
            box_xyxy_to_cxcywh(tgt_bbox), box_xyxy_to_cxcywh(anchors), p=1)# [t, num_anchors]
        cost_bbox = tf_cdist(
            box_xyxy_to_cxcywh(tgt_bbox), box_xyxy_to_cxcywh(pred_box), p=1)# [t, num_anchors]
        C = cost_bbox# [t, num_anchors]
        _, index = tf.nn.top_k(-C, k=self.pred_match_times)#[t, self.match_times]
        index = tf.transpose(index)#[self.match_times,t]
        C1 = cost_bbox_anchors# [t, num_anchors]
        _, index1 = tf.nn.top_k(-C1, k=self.match_times)#[t, self.match_times]
        index1 = tf.transpose(index1)#[self.match_times,t]
        indices = tf.concat([index, index1], axis=1)#[self.match_times,2*t]
        indices = tf.reshape(indices, [-1])

        pred_overlaps = iou_tf(tgt_bbox, pred_box)#[num_anchors, t]
        anchor_overlaps = iou_tf(tgt_bbox, anchors)#[num_anchors, t]

        pred_overlaps = tf.reduce_max(pred_overlaps, axis=1, keepdims=False)#[num_anchors,]
        pred_max_overlaps = tf.reduce_max(anchor_overlaps, axis=0, keepdims=False)#[t,]

        assigned_gt_inds, matched_gt_boxes = tf.py_func(self.py_,
                [pred_overlaps, pred_max_overlaps, anchor_overlaps, indices, gt_labels, tgt_bbox],
                [tf.int32, tf.float32]
            )
        assigned_gt_inds = tf.reshape(assigned_gt_inds, [-1])
        pos_inds = tf.where(tf.greater(assigned_gt_inds, 0))
        pos_inds = tf.to_int32(tf.reshape(pos_inds, [-1]))

        non_pos_inds = tf.where(tf.less_equal(assigned_gt_inds, 0))
        non_pos_inds = tf.to_int32(tf.squeeze(non_pos_inds))
        nonpos_num = tf.size(non_pos_inds)
        nonpos_reg_targets = tf.zeros([nonpos_num, 4], dtype=tf.float32)
        N = tf.shape(pos_inds)[0]

        neg_inds = tf.where(tf.equal(assigned_gt_inds, 0))
        neg_inds = tf.to_int32(tf.squeeze(neg_inds))

        pos_anchor = tf.gather(anchors, pos_inds)
        matches = assigned_gt_inds
        cls_targets = assigned_gt_inds
        if not cfg.giou_loss:
            pos_reg_targets = regress_target_tf(matched_gt_boxes, pos_anchor, img_size)
            reg_targets = tf.cond(tf.greater(N, 0),
                lambda:self.stitch(pos_inds, non_pos_inds, pos_reg_targets, nonpos_reg_targets), lambda:nonpos_reg_targets)
            return reg_targets, cls_targets, matches
        else:
            non_pos_gt = tf.zeros([nonpos_num, 4], dtype=tf.float32)
            # matched_gt_boxes, non_pos_gt = tf.py_func(self.show, 
            #     [matched_gt_boxes, non_pos_gt],
            #     [tf.float32, tf.float32]
            #     )

            gt_target = tf.cond(tf.greater(N, 0),
                lambda:self.stitch(pos_inds, non_pos_inds, matched_gt_boxes, non_pos_gt), lambda:non_pos_gt)
            # gt_target, non_pos_gt = tf.py_func(self.show, 
            #     [gt_target, non_pos_gt],
            #     [tf.float32, tf.float32]
            #     )    
            return gt_target, cls_targets, matches
    def stitch(self, pos_inds, non_pos_inds, pos_reg_targets, nonpos_reg_targets):
        reg_targets = tf.dynamic_stitch(
                [pos_inds, non_pos_inds],
                [pos_reg_targets, nonpos_reg_targets]
            )
        return reg_targets
    
    def show(self, p1,p2):
        p1 += p1
        p1 = p1/2
        p2 += p2
        p2 = p2/2
        print(p1.shape, p2.shape)
        return p1,p2

    def py_(self, pred_overlaps, pred_max_overlaps, anchor_overlaps, indexes, gt_labels, gt_box):
        '''
        
        '''
        gt_num = pred_max_overlaps.shape[0]
        anchor_num = anchor_overlaps.shape[0]

        assigned_gt_inds = np.zeros(anchor_num, dtype=np.int32)
        # print('pred_overlaps', pred_overlaps.shape)
        ignore_idx = np.where(pred_overlaps>self.neg_ignor_th)[0]
        # print('ignore_idx', ignore_idx.shape)
        assigned_gt_inds[ignore_idx] = -1
        pos_gt_index = np.arange(gt_num)
        pos_gt_index = np.tile(pos_gt_index, self.match_times+self.pred_match_times)
        pos_ious = anchor_overlaps[indexes, pos_gt_index]
        pos_ignore_idx = pos_ious < self.pos_ignor_th
        pos_gt_index_with_ignore = pos_gt_index + 1
        pos_gt_index_with_ignore[pos_ignore_idx] = -1
        assigned_gt_inds[indexes] = pos_gt_index_with_ignore
        pos_inds = np.where(assigned_gt_inds>0)[0]
        pos_gtbox = gt_box[assigned_gt_inds[pos_inds] - 1]
        assigned_gt_inds[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        # print(np.where(assigned_gt_inds>0)[0].shape, np.where(assigned_gt_inds<=0)[0].shape)
        return assigned_gt_inds, pos_gtbox