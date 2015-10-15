#!/bin/bash

TOOLS=../../build/tools


GLOG_logtostderr=1  $TOOLS/caffe train -solver lstm_solver_flow.prototxt -weights /nfs.yoda/xiaolonw/models/ai2_vgg16_flow_zero/video__iter_30000.caffemodel 

echo "Done."
