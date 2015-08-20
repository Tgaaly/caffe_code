# caffe_code

## Python API
feature_extraction.py -- python API for caffe

## Create loss layer in Caffe

1. Create the implementation in caffe/src/ (example seen below)
2. Add the class declaration to caffe/include/loss_layers.hpp

```
template <typename Dtype>
class TareksLossLayer : public LossLayer<Dtype> {
 public:
  explicit TareksLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "TareksLoss"; }
  /**
   * Unlike most loss layers, in the TareksLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
};
```

and then add the definitions in tareks_loss_layer.cpp and tareks_loss_layer.cu as follows:

```
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void TareksLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void TareksLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  Dtype loss = dot / bottom[0]->num() / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss*2; //multiplied by 2 here (after copying from EuclideanLossLayer)
}

template <typename Dtype>
void TareksLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
      caffe_cpu_axpby(
          bottom[i]->count(),              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(TareksLossLayer);
#endif

INSTANTIATE_CLASS(TareksLossLayer);
REGISTER_LAYER_CLASS(TareksLoss);

}  // namespace caffe
```

and then from caffe_root directory execute "make all"


## C++ API

---
title: CaffeNet C++ Classification example
description: A simple example performing image classification using the low-level C++ API.
category: example
include_in_docs: true
priority: 10
---

# Classifying ImageNet: using the C++ API

Caffe, at its core, is written in C++. It is possible to use the C++
API of Caffe to implement an image classification application similar
to the Python code presented in one of the Notebook example. To look
at a more general-purpose example of the Caffe C++ API, you should
study the source code of the command line tool `caffe` in `tools/caffe.cpp`.

## Presentation

A simple C++ code is proposed in
`examples/cpp_classification/classification.cpp`. For the sake of
simplicity, this example does not support oversampling of a single
sample nor batching of multiple independant samples. This example is
not trying to reach the maximum possible classification throughput on
a system, but special care was given to avoid unnecessary
pessimization while keeping the code readable.

## Compiling

The C++ example is built automatically when compiling Caffe. To
compile Caffe you should follow the documented instructions. The
classification example will be built as `examples/classification.bin`
in your build directory.

## Usage

To use the pre-trained CaffeNet model with the classification example,
you need to download it from the "Model Zoo" using the following
script:
```
./scripts/download_model_binary.py models/bvlc_reference_caffenet
```
The ImageNet labels file (also called the *synset file*) is also
required in order to map a prediction to the name of the class:
```
./data/ilsvrc12/get_ilsvrc_aux.sh.
```
Using the files that were downloaded, we can classify the provided cat
image (`examples/images/cat.jpg`) using this command:
```
./build/examples/cpp_classification/classification.bin \
  models/bvlc_reference_caffenet/deploy.prototxt \
  models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel \
  data/ilsvrc12/imagenet_mean.binaryproto \
  data/ilsvrc12/synset_words.txt \
  examples/images/cat.jpg
```
The output should look like this:
```
---------- Prediction for examples/images/cat.jpg ----------
0.3134 - "n02123045 tabby, tabby cat"
0.2380 - "n02123159 tiger cat"
0.1235 - "n02124075 Egyptian cat"
0.1003 - "n02119022 red fox, Vulpes vulpes"
0.0715 - "n02127052 lynx, catamount"
```

## Improving Performance

To further improve performance, you will need to leverage the GPU
more, here are some guidelines:

* Move the data on the GPU early and perform all preprocessing
operations there.
* If you have many images to classify simultaneously, you should use
batching (independent images are classified in a single forward pass).
* Use multiple classification threads to ensure the GPU is always fully
utilized and not waiting for an I/O blocked CPU thread.
