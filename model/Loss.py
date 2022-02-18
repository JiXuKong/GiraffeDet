import tensorflow as tf

from model.tool.regress_target import reverse_regress_target_tf#, normolize_box, reverse_normolize_box
from model.tool.uniform_matcher import OneImgUniformMatcher
from model.tool.losses import localization_loss, balanced_l1_loss, G_IOU_loss, focal_loss
import config as cfg


class Loss(object):
    def __init__(self, boxes_labels, num_boxes, pred, anchor, img_size):
        '''
        boxes_labels: batch, n, 5]
        num_boxes: [batch,]
        pred: list:
                pred[0] pred_class [b, h, w, class_num]
                pred[1] pred_delta [b, h, w, num_anchor*4]
        anchor:[-1, 4]
        '''
        self.boboxes_labelsxes=boxes_labels
        self.num_boxes=num_boxes
        b, self.class_num=tf.shape(pred[0])[0], tf.shape(pred[0])[2]
        self.pred_class=tf.reshape(pred[0], [b, -1, self.class_num])
        self.pred_delta=tf.reshape(pred[1], [b, -1, 4])
        self.anchor=anchor
        self.img_size=img_size
        self.class_weight=1.
        self.regress_weight=1.

    
    def batch_target(self):
        def fn(x):
            boxes, labels, num_boxes, pred_delta = x
            boxes, labels = boxes[:num_boxes], labels[:num_boxes]
            # if cfg.giou_loss:
            #     pred_boxes = reverse_normolize_box(pred_delta, self.img_size)
            # else:
            pred_boxes = reverse_regress_target_tf(pred_delta, self.anchor, self.img_size)
            pred_boxes = tf.reshape(pred_boxes, [-1, 4])
            reg_targets, cls_targets, matches = OneImgUniformMatcher(match_times=cfg.match_times).forward(
                self.anchor, [boxes, labels], pred_boxes, self.img_size
            )
            return pred_boxes, reg_targets, cls_targets, matches

        with tf.name_scope('target_creation'):

            pred_boxes, reg_targets, cls_targets, matches = tf.map_fn(
                fn, [self.boboxes_labelsxes[:, :, 1:], self.boboxes_labelsxes[:, :, 0], self.num_boxes, self.pred_delta],
                dtype=(tf.float32, tf.float32, tf.int32, tf.int32),
                parallel_iterations=4,
                back_prop=False, swap_memory=False, infer_shape=True
            )
            return pred_boxes, reg_targets, cls_targets, matches
    # def show(self, p1, p2):#for visualize data shape
    #     print(p1.shape, p2.shape)
    #     return p1, p2
    def _loss(self):
        pred_boxes, reg_targets, cls_targets, matches = self.batch_target()
        with tf.name_scope('losses'):
            # whether anchor is matched
            weights = tf.to_float(tf.greater(matches, 0))
            with tf.name_scope('classification_loss'):
                class_predictions = tf.identity(self.pred_class)
                # shape [batch_size, num_anchors, num_classes]
                cls_targets = tf.one_hot(cls_targets, self.class_num+1, axis=2)
                # shape [batch_size, num_anchors, num_classes + 1]
                # remove background
                cls_targets = tf.to_float(cls_targets[:, :, 1:])
                # now background represented by all zeros
                not_ignore = tf.to_float(tf.greater(matches, -1))
                # if a value is `-2` then we ignore its anchor
                cls_losses = focal_loss(
                    class_predictions, cls_targets, weights=not_ignore,
                    gamma=cfg.gama, alpha=cfg.alpha)
                # it has shape [batch_size, num_anchors]
            with tf.name_scope('localization_loss'):
                if not cfg.giou_loss:
                    encoded_boxes = tf.identity(self.pred_delta)
                    loc_losses = localization_loss(encoded_boxes, reg_targets, weights)
                else:
                    # pred_boxes, reg_targets = tf.py_func(self.show, #for visualize data shape
                    #     [pred_boxes, reg_targets],
                    #     [tf.float32, tf.float32]
                    # ) 
                    loc_losses = G_IOU_loss(pred_boxes, reg_targets, weights)
            with tf.name_scope('normalization'):
                matches_per_image = tf.reduce_sum(weights, axis=1)  # shape [batch_size]
                num_matches = tf.reduce_sum(matches_per_image, axis=0)  # shape []
                normalizer = tf.maximum(num_matches, 1.0)    
        
            loc_loss = tf.reduce_sum(loc_losses, axis=[0, 1])/normalizer
            cls_loss = tf.reduce_sum(cls_losses, axis=[0, 1])/normalizer
            #使用tf.loss
            tf.losses.add_loss(self.class_weight*cls_loss)
            tf.losses.add_loss(self.regress_weight*loc_loss)
            # add l2 regularization
            with tf.name_scope('weight_decay'):
                slim_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

                regularization_loss = tf.losses.get_regularization_loss()
            
            total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
        
        return total_loss, regularization_loss, loc_loss, cls_loss, normalizer, matches
        

