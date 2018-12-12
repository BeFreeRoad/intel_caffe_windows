# Intel Caffe Windows Inference
This fork Intel Caffe and Mini Caffe is dedicated to improving Caffe performance on Windows.

## Building
It tests on Microsoft Visual C++ 15.0 (Visual Studio 2017)

## Running
Current implementation uses OpenMP threads. By default the number of OpenMP threads is set
to the number of CPU cores. Each one thread is bound to a single core to achieve best
performance results. It is however possible to use own configuration by providing right
one through OpenMP environmental variables like OMP_NUM_THREADS.

## License and Citation
Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

***

