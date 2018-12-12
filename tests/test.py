#coding:utf-8
import sys
sys.path.append("/home/longriyao/test/caffe_normal/python")

import caffe
import numpy as np
import time


input_data = np.ones((3,112,112),dtype=np.float32)
net = caffe.Net("../model/hh.prototxt-bak", "../model/hh.caffemodel", caffe.TEST)
#net = caffe.Net("../test_bn/bn.prototxt", "../test_bn/bn.caffemodel", caffe.TEST)
#net = caffe.Net("../test_bn/hh.prototxt", "../test_bn/hh.caffemodel", caffe.TEST)
net.blobs['data'].data[0, :, :, :] = input_data
# 执行测试
#for i in xrange(100):
out = net.forward()
out=net.blobs['conv_1_batchnorm'].data
result_file = open("result.txt", "w")
for n in xrange(out.shape[0]):
    for c in xrange(out.shape[1]):
        for h in xrange(out.shape[2]):
            for w in xrange(out.shape[3]):
                result_file.write(str(out[n, c, h, w])+"\n")



