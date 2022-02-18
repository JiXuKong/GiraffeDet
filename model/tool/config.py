import numpy as np

#some file path

ckecpoint_file = './checkpoint'

train_img_txt = r'F:\open_dataset\VOCdevkit\0712train\07train.txt'
train_img_path = r'F:\open_dataset\VOCdevkit\0712train\JPEGImages'
train_label_path = r'F:\open_dataset\VOCdevkit\0712train\Annotations'

with open(train_img_txt, 'r') as f:
    image_index = [x.strip() for x in f.readlines()]
train_num = len(image_index)

test_img_txt = r'F:\open_dataset\VOCdevkit\07test\test.txt'
test_img_path = r'F:\open_dataset\VOCdevkit\07test\JPEGImages'
test_label_path = r'F:\open_dataset\VOCdevkit\07test\Annotations'


ssl_img_txt = 'train_ssl.txt'
ssl_img_path = ''
ssl_label_path = ''

cache_path = './pkl'

val_restore_path = './checkpoint/model.ckpt-11000'
train_restore_path = './pretrained/resnet_v1_50_2016_08_28/resnet_v1_50.ckpt'

#data parameter
classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

random_crop = True
other_aug = True
multiscale = False
class_to_ind = 0


#some network parameters
weight_decay = 0.0001
gradient_clip_by_norm = 10
batch_size = 4#测试时改为1
max_epoch = 1
with open(test_img_txt, 'r') as f:
    image_index = [x.strip() for x in f.readlines()]
test_num = len(image_index)
# test_num = 341//batch_size
class_num = len(classes)
image_size = 896#currently only 640
feature_size = [[image_size//(2**i), image_size//(2**i)] for i in range(3,8)]
phase = True
# base_anchor = [32, 64, 128]
base_anchor = [16]
scale = np.array([1, 2, 4, 8, 16, 32])
# scale = np.array([1])
aspect_ratio = np.array([1.0])
anchors = scale.shape[0]*aspect_ratio.shape[0]
momentum_rate = 0.9
alpha = 0.25
gama = 2
class_weight = 1.0
regress_weight = 1.0
decay = 0.99
pi = 1e-2

#yolof 
LR = 0.12/64*batch_size#0.01/(16/batch_size)
DECAY_STEP = [train_num//batch_size*20, train_num//batch_size*27]
score_threshold=0.05
nms_iou_threshold=0.3
encoder_channels=512
block_mid_channels=128
num_residual_blocks=7
block_dilations=[2, 3, 4, 5, 6, 7, 8]
cls_num_convs=2
reg_num_convs=4
match_times=4
max_detection_boxes_num=50