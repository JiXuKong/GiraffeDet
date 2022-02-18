from model.tool.regress_target import reverse_regress_target_tf
from model.YOLOF import YOLOF, DetectHead
from model.tool.timer import Timer
import config as cfg
# from model.tool.NMS import gpu_nms

from tensorflow.python.framework import graph_util
import tensorflow as tf

restore_path = r'F:\back_up\RetinaNet\train_less\other89_9\weights\model.ckpt-8000'

def build_graph(batch_size = 1, class_num = 7, image_size = 512):
    input_ = tf.placeholder(tf.float32, shape = [1, image_size, image_size, 3])
    out, all_anchors = YOLOF(base_anchor=cfg.base_anchor,
        scale=cfg.scale, aspect_ratio=cfg.aspect_ratio, class_num=len(cfg.classes)-1, is_training=False).forward(input_)
    _cls_scores, _cls_classes, _boxes = DetectHead(cfg.score_threshold, cfg.nms_iou_threshold,
        cfg.max_detection_boxes_num, ).forward(inputs_0=tf.expand_dims(out[0], axis=0), inputs_1=tf.expand_dims(out[1], axis=0), 
        anchor=all_anchors, 
        imgsize=cfg.image_size, 
        class_num=len(cfg.classes)-1,
        score_thresh=cfg.score_threshold,
        nms_thresh=cfg.nms_iou_threshold)
    boxes, scores, label = _boxes[0], _cls_scores[0], _cls_classes[0]#for one img

    boxes = tf.concat(boxes, axis = 0, name = 'boxes')
    scores = tf.concat(scores, axis = 0, name = 'scores')
    labels = tf.concat(label, axis = 0, name = 'label')
    
    return boxes, scores, labels

pb_file_path = r'F:\github\YOLOF_TF\pb'
with tf.Session(graph=tf.Graph()) as sess:
    boxes, scores, labels = build_graph()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, restore_path)
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['boxes', 'scores', 'label'])
    with tf.gfile.FastGFile(pb_file_path+'/YOLOF_res50_Frozen.pb', mode='wb') as f:
        f.write(constant_graph.SerializeToString())
        print(pb_file_path+'/YOLOF_res50_Frozen.pb')

