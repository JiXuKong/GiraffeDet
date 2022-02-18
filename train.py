from tensorflow.python import pywrap_tensorflow
import tensorflow as tf
import numpy as np
import os, sys

from model.tool import xx1
from data.pascal_voc import pascal_voc
from model.tool.timer import Timer
from model.GiraffeDet import GiraffeDet, DetectHead
from model.tool.learning_schedules import cosine_decay_with_warmup
from model.Loss import Loss
import config as cfg 


slim = tf.contrib.slim

data = pascal_voc('train', False, cfg.train_img_path, cfg.train_label_path, cfg.train_img_txt, True)


input_ = tf.placeholder(tf.float32, shape = [cfg.batch_size, cfg.image_size, cfg.image_size, 3])
get_boxes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80, 5])
num_boxes = tf.placeholder(tf.int32, shape = [cfg.batch_size,])
imsize = tf.placeholder(tf.float32, shape = [None,])
# get_classes = tf.placeholder(tf.float32, shape = [cfg.batch_size, 80])


out, all_anchors = GiraffeDet(base_anchor=cfg.base_anchor,
 scale=cfg.scale, aspect_ratio=cfg.aspect_ratio, class_num=len(cfg.classes)-1, is_training=True).forward(input_)

g = tf.get_default_graph()

# print('out', out)
total_loss, regularization_loss, loc_loss, cls_loss, normalizer, matches = Loss(get_boxes, num_boxes,
 out, all_anchors, imsize)._loss()  
with tf.name_scope('class_loss'):    
    tf.losses.add_loss(cls_loss*cfg.class_weight)

with tf.name_scope('reg_loss'):     
    tf.losses.add_loss(loc_loss*cfg.regress_weight)
with tf.name_scope('total_loss'):     
    total_loss = total_loss

with tf.name_scope('detection'): 
    # print('out', out)
    _cls_scores, _cls_classes, _boxes = DetectHead(cfg.score_threshold, cfg.nms_iou_threshold,
     cfg.max_detection_boxes_num, ).forward(inputs_0=tf.expand_dims(out[0][0], axis=0), inputs_1=tf.expand_dims(out[1][0], axis=0), 
     anchor=all_anchors, 
     imgsize=cfg.image_size, 
     class_num=len(cfg.classes)-1,
     score_thresh=0.3,
     nms_thresh=cfg.nms_iou_threshold)
    # print('_boxes', _boxes)
    
    nms_box, nms_score, nms_label = tf.reshape(_boxes, [-1, 4]), tf.reshape(_cls_scores, [-1,]), tf.reshape(_cls_classes, [-1,])
    # print('tf.expand_dims(input_[0], 0)', tf.expand_dims(input_[0], 0))
    # print('nms_box', nms_box)
    # print('nms_label+1', nms_label+1)
    # print("nms_score", nms_score)
    detection_in_img = xx1.draw_boxes_with_categories_and_scores(img_batch=tf.expand_dims(input_[0], 0),
                                                             boxes=nms_box,
                                                             labels=nms_label+1,
                                                             scores=nms_score)
    gt_in_img = xx1.draw_boxes_with_categories_and_scores(img_batch=tf.expand_dims(input_[0], 0),
                                                             boxes=get_boxes[0, :, 1:],
                                                             labels=get_boxes[0, :, 0],
                                                             scores=get_boxes[0, :, 0])
    total_anchor = tf.reshape(all_anchors, [-1,4])
    # print(matches)
    match = matches[0]
    positive_anchor = tf.gather(total_anchor, tf.reshape(tf.to_int32(tf.where(tf.greater(match, 0))), [-1]))
    # temp_inds = tf.to_int32(tf.greater(match, 0))
    # positive_anchor = total_anchor
    # ignore_anchor = tf.gather(total_anchor, tf.reshape(tf.less(match, 0), [-1]))
    pos_labels = tf.gather(match, tf.reshape(tf.to_int32(tf.where(tf.greater(match, 0))), [-1]))
    # ignore_labels = tf.ones(tf.shape(positive_anchor)[0])
    sample_in_img = xx1.draw_boxes_with_categories(img_batch=tf.expand_dims(input_[0], 0),
                                                boxes=positive_anchor,
                                                labels=pos_labels)
    # print('tf.expand_dims(input_[0], 0)', tf.expand_dims(input_[0], 0))
    # print('get_boxes[0, :, 1:]', get_boxes[0, :, 1:])
    # print('get_boxes[0, :, 0]', get_boxes[0, :, 0])
    tf.summary.image('detection', detection_in_img)
    tf.summary.image('GT', gt_in_img)
    tf.summary.image('pos_sample', sample_in_img)
tf.contrib.quantize.create_training_graph(g, 45924)
global_step = slim.get_or_create_global_step()
max_epoch = 55
epoch_step = int(cfg.train_num//cfg.batch_size)
with tf.variable_scope('learning_rate'):
    # lr = cosine_decay_with_warmup(
    #     global_step = global_step,
    #     learning_rate_base = cfg.LR,
    #     total_steps = int(epoch_step*max_epoch),
    #     warmup_learning_rate=0.000066667/64*cfg.batch_size,
    #     warmup_steps=int(epoch_step*2))
# with tf.variable_scope('learning_rate'):
    lr_ = tf.train.piecewise_constant(global_step,
                                     boundaries=[np.int64(cfg.DECAY_STEP[0]), np.int64(cfg.DECAY_STEP[1])],
                                     values=[cfg.LR, cfg.LR / 10., cfg.LR / 100.])
    def warm_up(init_lr, max_lr, warm_step, global_step):
        return init_lr + (max_lr - init_lr)/float(warm_step)*tf.cast(global_step, tf.float32)
    warm_step = 2000
    init_lr = 1e-6
    max_lr = cfg.LR
    lr = tf.where(global_step<warm_step, warm_up(init_lr, max_lr, warm_step, global_step), lr_)
#     LR = 1e-5
#     lr = tf.train.piecewise_constant(global_step,
#                                      boundaries=[np.int64(1.01e5), np.int64(1.02e5), np.int64(1.07e5), np.int64(1.12e5)],
#                                      values=[LR, LR * 10., LR * 100., LR * 10., LR])

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    # optimizer = tf.train.MomentumOptimizer(learning_rate, momentum_rate, use_nesterov=False)
    optimizer = tf.train.MomentumOptimizer(lr, cfg.momentum_rate, use_nesterov=False)
    # optimizer = tf.train.AdamOptimizer(lr)
    gradient = optimizer.compute_gradients(total_loss)
    with tf.name_scope('clip_gradients_YJR'):
        gradient = slim.learning.clip_gradient_norms(gradient,cfg.gradient_clip_by_norm)

    with tf.name_scope('apply_gradients'):
        train_op = optimizer.apply_gradients(grads_and_vars=gradient,global_step=global_step)
# train_op = tf.train.experimental.enable_mixed_precision_graph_rewrite(train_op)
g_list = tf.global_variables()

total_params = 0
for v in tf.trainable_variables():
    shape = v.get_shape()
    cnt = 1
    for dim in shape:
        cnt *= dim.value
    total_params += cnt
    # total_params += cnt
print('total_params', str(total_params/1e6) + 'M')

# for g in g_list:
#     print(g.name)AdamMomentum
save_list = [g for g in g_list if ('Momentum' not in g.name)and('ExponentialMovingAverage' not in g.name)]
saver = tf.train.Saver(var_list=save_list, max_to_keep=30)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
summary_op = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(cfg.ckecpoint_file, sess.graph)


def get_variables_in_checkpoint_file(file_name):
    reader = pywrap_tensorflow.NewCheckpointReader(file_name)
    var_to_shape_map = reader.get_variable_to_shape_map()
    return var_to_shape_map

def initialize(pretrained_model, variable_to_restore):
    var_keep_dic = get_variables_in_checkpoint_file(pretrained_model)
    # Get the variables to restore, ignoring the variables to fix
    variables_to_restore = get_variables_to_restore(variable_to_restore, var_keep_dic)
    restorer = tf.train.Saver(variables_to_restore)
    return restorer

def get_variables_to_restore(variables, var_keep_dic):
    variables_to_restore = []

    for v in variables:
        if (v.name == 'global_step:0'):
            continue

#         if(v.name.split('/')[1] != 'ClassPredictor')\
#         and(v.name.split('/')[1] != 'BoxPredictor')\
#         and(v.name.split(':')[0])in var_keep_dic:
#         if 'WeightSharedConvolutionalBoxPredictor' not in v.name\
#          and 'FeatureExtractor' not in v.name\
        
        if (v.name.split(':')[0])in var_keep_dic:

            # print('Variables restored: %s' % v.name)
            variables_to_restore.append(v)
        
    return variables_to_restore

if cfg.train_restore_path is not None:
    print('Restoring weights from: ' + cfg.train_restore_path)
    # restorer = initialize(cfg.train_restore_path, g_list)
    restorer = tf.train.Saver(save_list)
    restorer.restore(sess, cfg.train_restore_path)
    

def train():
    total_timer = Timer()
    train_timer = Timer()
    load_timer = Timer()
    max_epoch = 55
    epoch_step = int(cfg.train_num//cfg.batch_size)
    t = 1
    for epoch in range(1, max_epoch + 1):
        print('-'*25, 'epoch', epoch,'/',str(max_epoch), '-'*25)


        t_loss = 0
        ll_loss = 0
        r_loss = 0
        c_loss = 0
        
        
       
        for step in range(1, epoch_step + 1):
            if step == epoch == 0:
                print('trainable_params', str(total_params/1e6) + 'M')
     
            t = t + 1
            total_timer.tic()
            load_timer.tic()
 
            images, labels, imnm, num_boxes_, imsize_ = data.get()
            # print(imsize, imsize.dtype)
            
#             load_timer.toc()
            feed_dict = {input_: images,
                         get_boxes: labels,
                         num_boxes: num_boxes_,
                         imsize:imsize_
                        }
            
            _, g_step_, total_loss_, r_loss, loc_loss_, cls_loss_, lr_ = sess.run(
                [train_op,
                    global_step,
                    total_loss,
                    regularization_loss,
                    loc_loss, 
                    cls_loss,
                    lr], feed_dict = feed_dict)
            # print(m.shape, m)
            # print(m1.shape, m1)
            total_timer.toc()
            if g_step_%50 ==0:
                sys.stdout.write('\r>> ' + 'iters '+str(g_step_)+str('/')+str(epoch_step*max_epoch)+' loss '+str(total_loss_) + ' ')
                sys.stdout.flush()
                summary_str = sess.run(summary_op, feed_dict = feed_dict)
                
                train_total_summary = tf.Summary(value=[
                    tf.Summary.Value(tag="config/learning rate", simple_value=lr_),
                    tf.Summary.Value(tag="train/classification/focal_loss", simple_value=cfg.class_weight*cls_loss_),
                    # tf.Summary.Value(tag="train/classification/cnt_loss", simple_value=cfg.cnt_weight*cn_loss),
#                     tf.Summary.Value(tag="train/p_nm", simple_value=p_nm_),
                    tf.Summary.Value(tag="train/regress_loss", simple_value=cfg.regress_weight*loc_loss_),
#                     tf.Summary.Value(tag="train/clone_loss", simple_value=cfg.class_weight*cl_loss + cfg.regress_weight*re_loss + cfg.cnt_weight*cn_loss),
                    tf.Summary.Value(tag="train/l2_loss", simple_value=r_loss),
                    tf.Summary.Value(tag="train/total_loss", simple_value=total_loss_)
                    ])
                print('curent speed: ', total_timer.diff, 'remain time: ', total_timer.remain(g_step_, epoch_step*max_epoch))
                summary_writer.add_summary(summary_str, g_step_)
                summary_writer.add_summary(train_total_summary, g_step_)
            if g_step_%5000 == 0:
                print('saving checkpoint')
                saver.save(sess, cfg.ckecpoint_file + '/model.ckpt', g_step_)

        total_timer.toc()
        sys.stdout.write('\n')
        print('>> mean loss', t_loss)
        print('curent speed: ', total_timer.average_time, 'remain time: ', total_timer.remain(g_step_, epoch_step*max_epoch))
        
    print('saving checkpoint')
    saver.save(sess, cfg.ckecpoint_file + '/model.ckpt', g_step_)
    
    
train()