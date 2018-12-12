#coding:utf-8
import sys
sys.path.append("/home/longriyao/test/caffe_normal/python")

import caffe.proto.caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import caffe
import numpy as np
import time

net_file = "../model/hh.prototxt"
weight_file = "../model/hh.caffemodel"
net = caffe_pb2.NetParameter()
text_format.Merge(open(net_file, 'r').read(), net)
weight = caffe_pb2.NetParameter()
weight.ParseFromString(open(weight_file, 'rb').read())


for layer in weight.layer:
    if layer.type == "Convolution":
        print layer.convolution_param.bias_term



