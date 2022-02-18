import numpy as np



    
def voc_ap(rec, prec, use_07_metric=False):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_eval(gt_dict, val_preds, classidx, iou_thres=0.5, use_07_metric=False):
    '''
    Top level function that does the PASCAL VOC evaluation.
    '''
    #gt_dict:[{imgid:[x1,y1,x2,y2, class_id]}]
    # 1.obtain gt: extract all gt objects for this class
    class_recs = {}
    npos = 0
    for img_id in gt_dict:
        R = [obj for obj in gt_dict[img_id] if obj[-1] == classidx]
        bbox = np.array([x[:4] for x in R])
        det = [False] * len(R)
        npos += len(R)
        class_recs[img_id] = {'bbox': bbox, 'det': det}
    #pred:[img_ids, x1,y1,x2,y2, confidence, class_id]
    # 2. obtain pred results
    pred = [x for x in val_preds if x[-1] == classidx]
    img_ids = [x[0] for x in pred]
    confidence = np.array([x[-2] for x in pred])
    BB = np.array([[x[1], x[2], x[3], x[4]] for x in pred])

    # 3. sort by confidence
    sorted_ind = np.argsort(-confidence)
    try:
        BB = BB[sorted_ind, :]
    except:
        print('no box, ignore')
        return 1e-6, 1e-6, 0
    img_ids = [img_ids[x] for x in sorted_ind]

    # 4. mark TPs and FPs
    nd = len(img_ids)#总共预测框的个数
    tp = np.zeros(nd)
    fp = np.zeros(nd)

    for d in range(nd):
        # all the gt info in some image
        R = class_recs[img_ids[d]]#imgid的gt,一个id对应多个gt?
        bb = BB[d, :]#对应一个imgid的一个预测框
        ovmax = -np.Inf
        BBGT = R['bbox']#对应imgid的gt框

        if BBGT.size > 0:
            # calc iou
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + (BBGT[:, 2] - BBGT[:, 0] + 1.) * (
                        BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > iou_thres:
            # gt not matched yet
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)
#     print('tp: ', tp)
#     print('fp: ', fp)
#     print('tp+fn: ', npos)
    return rec, prec, ap
