#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
#import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse

CLASSES = ('__background__', 'face')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

g_fp = open('./tantan-fd-1000-rlt.txt', 'w+')

def print_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    print 'thresh in print_detection: {:.3f}'.format(thresh)

    inds = np.where(dets[:, -1] >= thresh)[0]
    print inds.shape

    if len(inds) == 0:
        return

#    im = im[:, :, (2, 1, 0)]
#    fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        
        print ('Detection {:d}: cls = {:s}, bbox = ({:.3f} {:.3f} {:.3f} {:.3f}), '
               'score = {:.3f}').format(i, class_name, bbox[0], bbox[1], bbox[2], bbox[3], score)
                
#        ax.add_patch(
#            plt.Rectangle((bbox[0], bbox[1]),
#                          bbox[2] - bbox[0],
#                          bbox[3] - bbox[1], fill=False,
#                          edgecolor='red', linewidth=3.5)
#            )
#        ax.text(bbox[0], bbox[1] - 2,
#                '{:s} {:.3f}'.format(class_name, score),
#                bbox=dict(facecolor='blue', alpha=0.5),
#                fontsize=14, color='white')
#
#    ax.set_title(('{} detections with '
#                  'p({} | box) >= {:.1f}').format(class_name, class_name,
#                                                  thresh),
#                  fontsize=14)
#    plt.axis('off')
#    plt.tight_layout()
#    plt.draw()


#def vis_detections(im, class_name, dets, thresh=0.5):
#    """Draw detected bounding boxes."""
#    inds = np.where(dets[:, -1] >= thresh)[0]
#    if len(inds) == 0:
#        return
#
#    im = im[:, :, (2, 1, 0)]
#    fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(im, aspect='equal')
#    for i in inds:
#        bbox = dets[i, :4]
#        score = dets[i, -1]
#
#        ax.add_patch(
#            plt.Rectangle((bbox[0], bbox[1]),
#                          bbox[2] - bbox[0],
#                          bbox[3] - bbox[1], fill=False,
#                          edgecolor='red', linewidth=3.5)
#            )
#        ax.text(bbox[0], bbox[1] - 2,
#                '{:s} {:.3f}'.format(class_name, score),
#                bbox=dict(facecolor='blue', alpha=0.5),
#                fontsize=14, color='white')
#
#    ax.set_title(('{} detections with '
#                  'p({} | box) >= {:.1f}').format(class_name, class_name,
#                                                  thresh),
#                  fontsize=14)
#    plt.axis('off')
#    plt.tight_layout()
#    plt.draw()

def save_detections_to_file(fp, im, class_name, dets, thresh=0.5):
    """Save detected bounding boxes into file."""
    if not fp:
        return

    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

#    im = im[:, :, (2, 1, 0)]
#    fig, ax = plt.subplots(figsize=(12, 12))
#    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        
        fp.write(('Detection {:d}: cls = {:s}, bbox = ({:.3f} {:.3f} {:.3f} {:.3f}), '
               'score = {:.3f}\n').format(i, class_name, bbox[0], bbox[1], bbox[2], bbox[3], score))

#def demo(net, image_name):
def demo(net, im_file, thresh=0.8, fp_rlt=None):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
#    im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
    im = cv2.imread(im_file)
    
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    if fp_rlt:
        fp_rlt.write(('Detection took {:.3f}s for '
                    '{:d} object proposals\n').format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    #CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    if thresh<0.0:
        thresh = 0.0
    if NMS_THRESH>thresh:
        NMS_THRESH = thresh

    print ('thresh={:.3f}, NMS_THRESH={:.3f}').format(thresh, NMS_THRESH)
    if fp_rlt:
        fp_rlt.write(('thresh={:.3f}, NMS_THRESH={:.3f}').format(thresh, NMS_THRESH))

    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        
        print ('cls = {:s}: {:d} detections before NMS').format(cls, dets.shape[0])
        if fp_rlt:
            fp_rlt.write(('cls = {:s}: {:d} detections before NMS').format(cls, dets.shape[0]))

        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]

        print ('cls = {:s}: {:d} detections after NMS').format(cls, dets.shape[0])
        if fp_rlt:
            fp_rlt.write(('cls = {:s}: {:d} detections after NMS').format(cls, dets.shape[0]))


	    #print_detections(im, cls, dets, thresh=CONF_THRESH)    
        print_detections(im, cls, dets, thresh)        
        #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        
        if dets is not None:
            g_fp.write(('{:s}\t{:d}\t{:.3f}\n').format( im_file, len(dets), np.max(dets[:, -1]) ))
        else:
            g_fp.write(('{:s}\t{:d}\t{:.3f}\n').format( im_file, 0, 0))

        if fp_rlt:
            #save_detections_to_file(fp, im, cls, dets, thresh=CONF_THRESH)
            save_detections_to_file(fp_rlt, im, cls, dets, thresh)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [zf]',
                        choices=NETS.keys(), default='zf')
#added by zhaoyafei 20160830
    parser.add_argument('--filelist', dest='file_list', help='file of image list [./list_img.txt]',
                        default='./list_img.txt')
    parser.add_argument('--output', dest='rlt_file', help='output file to save detection results [./det_rlt.txt]',
                        default='./det_rlt.txt')      
    parser.add_argument('--thresh', dest='thresh', help='threshold for detection confidence [0.5]',
                        default=0.8, type=float)                                           
#end of add

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    fp_rlt = open(args.rlt_file, "w+")

    fp_rlt.write('\n\nLoaded network {:s}\n'.format(caffemodel))
    #read images from image list file, one image per line
    if args.file_list:
        fp = open(args.file_list, "r")
        for line in fp:
            im_name = line.strip()
            
            if im_name.startswith('.') or im_name.startswith('data'):
                im_file = os.path.join(cfg.ROOT_DIR, im_name)
            else:
                im_file = im_name

            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo for ' + im_file
            fp_rlt.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            fp_rlt.write('Demo for ' + im_file + '\n')
            demo(net, im_file, args.thresh, fp_rlt)
            fp_rlt.write('----->\n\n')

    else:#read from demo/data/*
        im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                    '001763.jpg', '004545.jpg']
        for im_name in im_names:
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            print 'Demo for data/demo/{}'.format(im_name)
            im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
            fp_rlt.write('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
            fp_rlt.write('Demo for ' + im_file + '\n')
            demo(net, im_file, args.thresh, fp_rlt)
            fp_rlt.write('----->\n\n')

    fp_rlt.close()

    #plt.show()

g_fp.close()