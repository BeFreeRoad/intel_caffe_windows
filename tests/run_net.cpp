#include <random>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <thread>

#include <caffe/net.hpp>

using namespace std;
using namespace caffe;

int main(int argc, char *argv[]) {

  auto net = new caffe::Net(argv[1]);
  net->CopyTrainedLayersFrom(argv[2]);

  std::shared_ptr<Blob> input = net->blob_by_name("data");
  for (int k = 0; k < input->count(); ++k) {
    input->mutable_cpu_data()[k] = 1;
  }
  // forward network
  auto st = std::chrono::high_resolution_clock::now();
  net->Forward();
  auto et = std::chrono::high_resolution_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds> (et - st).count();
  std::cout <<  "spend --- "<< ms << " ---ms" << std::endl;
  return 0;
}

