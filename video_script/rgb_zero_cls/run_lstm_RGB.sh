#!/bin/bash

TOOLS=../../build/tools


GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_RGB.prototxt  -weights /nfs.yoda/xiaolonw/models/ai2_vgg16_app_zero/video__iter_50000.caffemodel

  
echo "Done."
