import numpy as np
import sys
from operator import itemgetter
import matplotlib.pyplot as plt
'''
    mAP calculating tool by VOC12 method for object-detection.
    Build for model test or model validate.
'''


def get_data(input_path):
    """Parse the data from annotation file

    Args:
        input_path: annotation file path

    Returns:
        all_data: list(filepath, width, height, list(bboxes))
        classes_count: dict{key:class_name, value:count_num}
            e.g. {'Car': 2383, 'Mobile phone': 1108, 'Person': 3745}
        class_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    found_bg = False
    all_imgs = {}

    classes_count = {}

    class_mapping = {}

    i = 1

    with open(input_path, 'r') as f:

        print('Parsing annotation files')

        for line in f:

            # Print process
            sys.stdout.write('\r' + 'idx=' + str(i))
            i += 1

            # line_split = line.strip().split(',')

            # (filename, y1, x1, y2, x2, class_name) = line_split
            print(line)
            filename = line.split()[0]
            y1, x1, y2, x2, class_name = line.split()[1].split(',')

            if class_name == '':
                # wrong labelling
                continue

            if class_name not in classes_count:
                classes_count[class_name] = 1
            else:
                classes_count[class_name] += 1

            if class_name not in class_mapping:
                if class_name == 'bg' and found_bg == False:
                    print(
                        'Found class name bg.Will be treated as background (this is usually for hard negative mining).')
                    found_bg = True
                class_mapping[class_name] = len(class_mapping)

            if filename not in all_imgs:
                all_imgs[filename] = {}

                # img = cv2.imread(filename)
                # (rows, cols) = img.shape[:2]
                all_imgs[filename]['filepath'] = filename
                # all_imgs[filename]['width'] = cols
                # all_imgs[filename]['height'] = rows
                all_imgs[filename]['bboxes'] = []

            all_imgs[filename]['bboxes'].append(
                {'class': class_name, 'x1': int(x1), 'x2': int(x2), 'y1': int(y1), 'y2': int(y2)})

        all_data = []
        for key in all_imgs:
            all_data.append(all_imgs[key])

        # make sure the bg class is last in the list
        if found_bg:
            if class_mapping['bg'] != len(class_mapping) - 1:
                key_to_switch = [key for key in class_mapping.keys() if class_mapping[key] == len(class_mapping) - 1][0]
                val_to_switch = class_mapping['bg']
                class_mapping['bg'] = len(class_mapping) - 1
                class_mapping[key_to_switch] = val_to_switch

        return all_data, classes_count, class_mapping


def union(au, bu, area_intersection):
    area_a = (au[2] - au[0]) * (au[3] - au[1])
    area_b = (bu[2] - bu[0]) * (bu[3] - bu[1])
    area_union = area_a + area_b - area_intersection        # 区域并集
    return area_union


def intersection(ai, bi):
    x = max(ai[0], bi[0])
    y = max(ai[1], bi[1])
    w = min(ai[2], bi[2]) - x
    h = min(ai[3], bi[3]) - y       # 区域交集为(x,y,w,h)
    if w < 0 or h < 0:
        return 0
    return w*h


def iou(a, b):
    # a and b should be (x1,y1,x2,y2)
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    area_i = intersection(a, b)
    area_u = union(a, b, area_i)

    return float(area_i) / float(area_u + 1e-6)


def read_data(result_file, gt_file, iou_thr=0.5, mode='test', **kwargs):
    '''
        read result/ground truth data and register all bboxes by class, confidence, status
    :param mode: 'test' mode used in testing result file; 'validate' mode used in model validating
        , file format see config
    :param iou_thr: iou threshold of true-positive sample
    :param result_file: (img_name y1, x1, y2, x2, key:confidence)
    :param gt_file: (img_name y1, x1, y2, x2, key)
    :return:
        cls_record: dict, lists of bboxes
        conf: dict, (bbox_id:confidence)
        status: dict, (bbox_id:1/0)
    '''
    # read data and convert data to list of dicts
    if mode == 'test':
        result, _, _ = get_data(result_file)
        gt, cls_count, cls_mapping = get_data(gt_file)
    elif mode == 'validate':
        result = result_file
        gt = gt_file
        cls_count = kwargs['cls_count']['cls_count']         # parameters were passed twice
    else:
        raise NameError("Undefined mode '{}'".format(mode))

    gt_dict = {}        # convert gt data to dict form
    conf = {}   # bbox confidence{box:conf}
    status = {}     # bbox kind{box:1(TP)/0(FP)
    cls_record = {}     # store bbox class

    for img in gt:
        # convert ground truth data from list to dict
        gt_dict[img['filepath'].split('/')[-1]] = img

    for img in result:
        # check all images
        try:
            img_gt = gt_dict[img['filepath'].split('/')[-1]]['bboxes']      # gts of img
        except:
            # could not find ground truth data
            continue
        for bbox in img['bboxes']:
            # set bbox confidence and class
            bbox['class'], bbox['conf'] = bbox['class'].split(':')
            bbox['conf'] = float(bbox['conf'])
        # sort bboxes by confidence
        img_bboxes = sorted(img['bboxes'], key=itemgetter('conf'))
        img_bboxes.reverse()
        for idx, bbox in enumerate(img_bboxes):
            # check all bboxes in result
            bbox_id = img['filepath'].split('/')[-1]+str(idx)
            best_iou = 0
            try:
                # register bbox to its class
                cls_record[bbox['class']].append(bbox_id)
            except:
                cls_record[bbox['class']] = [bbox_id]
            conf[bbox_id] = bbox['conf']
            for idy, check_gt in enumerate(img_gt):
                # traverse all available gt for best match
                if check_gt['class'] != bbox['class']:
                    # different class
                    continue
                res_gt_iou = iou((bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']),
                                 (check_gt['x1'], check_gt['y1'], check_gt['x2'], check_gt['y2']))
                if res_gt_iou > best_iou:
                    best_iou = res_gt_iou
                    ref_gt = idy
            if best_iou < iou_thr:
                # false positive
                status[bbox_id] = 0
            else:
                # true positive
                status[bbox_id] = 1
                del img_gt[ref_gt]
    return cls_record, conf, status, cls_count


def mk_table(cls, cls_record, conf, status, cls_count):
    box_list = cls_record[cls]
    all_tru = cls_count[cls]        # TP+FN

    cls_table = np.zeros((len(box_list), 2))
    pre = list()        # precision
    rec = list()        # recall

    for idx, bbox in enumerate(box_list):
        cls_table[idx, 0] = conf[bbox]      # confidence
        cls_table[idx, 1] = status[bbox]        # status
    # sort by confidence
    index_lis = list(cls_table[:, 0].argsort())
    index_lis.reverse()
    cls_table = cls_table[index_lis]
    tp_samples = np.sum(cls_table[:, 1])
    for thr in range(1, cls_table.shape[0]+1):
        TP = np.sum(cls_table[:thr, 1])
        all_pos = thr
        precision = TP/all_pos
        recall = TP/all_tru
        pre.append(precision)
        rec.append(recall)
    return pre, rec, tp_samples


def voc_ap(rec, prec):
    """
        code borrow from https://github.com/Cartucho/mAP
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def voc_mAP(result_path, ground_truth_path, mode='test', **kwargs):
    '''
        calculate mAP by standard of voc12
    '''
    cls_ap = list()
    cls_rec = list()
    cls_pre = list()
    cls_tp_samples = dict()
    cls_record, conf, status, cls_count = read_data(result_path, ground_truth_path, mode=mode, cls_count=kwargs)
    for cls, cls_num in cls_count.items():
        # traverse all classes
        try:
            pre, rec, tp_samples = mk_table(cls, cls_record, conf, status, cls_count)
            ap, mrec, mpre = voc_ap(rec, pre)
        except:
            # no results of some classes
            ap = mrec = mpre = tp_samples = 0
        cls_ap.append(ap)
        cls_rec.append(mrec)
        cls_pre.append(mpre)
        cls_tp_samples[cls] = tp_samples
    mAP = sum(cls_ap)/len(cls_ap)
    return mAP, [cls_ap, cls_rec, cls_pre, cls_count, cls_record, cls_tp_samples]


# ------------------------------------- config -------------------------------------- #
# mAP calculator by standard of voc12
# 'test' mode input file format:
#       input: result file path, ground truth file path
#       result line: (img_name y1, x1, y2, x2, key:confidence)
#       ground truth line: (img_name y1, x1, y2, x2, key)
# 'validate' mode input format:
#       input: result, ground truth, class count of validation set
#       result: list of {img_name, bboxes:list of {y1, x1, y2, x2, key:confidence}}
#       ground truth: list of {img_name, bboxes: list of {y1, x1, y2, x2, key}}
if __name__ == '__main__':
    result_path = './data/res.txt'
    ground_truth_path = './data/Boats/test/test.txt'
    show_visual_result = False

    mAP, [cls_ap, cls_rec, cls_pre, cls_count, cls_record, cls_tp_samples] = \
        voc_mAP(result_path, ground_truth_path)
    # print('model validate result:')
    # print('mAP:{}\nAP of all classes:{}'.format(mAP, cls_ap))

    # show in figure
    # ap of each class
    if show_visual_result:

        cls_num = len(cls_count)
        plt.figure(1)
        plt.barh(range(cls_num), cls_ap, 0.3)
        plt.title('APs of all classes(mAP={:4f})'.format(mAP))
        plt.xlabel('AP')
        plt.yticks(range(cls_num), list(cls_count.keys()))
        for idx, ap in enumerate(cls_ap):
            plt.text(ap, idx-0.05, '{:.4f}'.format(ap))
        # tp samples of each class
        plt.figure(2)
        plt.title('result sample composition')
        cls_fp_samples = dict()
        for cls, count in cls_count.items():
            try:
                cls_fp_samples[cls] = len(cls_record[cls])-cls_tp_samples[cls]
            except:
                cls_fp_samples[cls] = 0
        cls_fp_samples = list(cls_fp_samples.values())      # number of fp samples
        cls_tp_samples = list(cls_tp_samples.values())      # number of tp samples
        fps = plt.bar(range(cls_num), cls_fp_samples, width=0.3, label='FP', bottom=cls_tp_samples)
        tps = plt.bar(range(cls_num), cls_tp_samples, width=0.3, label='TP')
        plt.xticks(range(cls_num), list(cls_count.keys()))
        for x, y in enumerate(cls_tp_samples):
            plt.text(x, y-0.15, '{:.0f}'.format(y))
        plt.legend()
        # ground truth
        plt.figure(3)
        plt.title('the number of ground-truths of each class')
        cls_gt = list(cls_count.values())
        plt.bar(range(cls_num), cls_gt, width=0.3, label='ground truth')
        plt.xticks(range(cls_num), list(cls_count.keys()))
        for x, y in enumerate(cls_gt):
            plt.text(x-0.05, y+0.15, '{:.0f}'.format(y))
        plt.show()

