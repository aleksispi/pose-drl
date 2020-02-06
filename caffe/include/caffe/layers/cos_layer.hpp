#ifndef CAFFE_COS_LAYER_HPP_
#define CAFFE_COS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/neuron_layer.hpp"

namespace caffe {

template <typename Dtype>
class CosLayer : public NeuronLayer<Dtype> {
  public:
    explicit CosLayer(const LayerParameter& param)
        : NeuronLayer<Dtype>(param) {}

    virtual inline const char* type() const { return "Cos"; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
       const vector<Blob<Dtype>*>& top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
       const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

     };

}  // namespace caffe

#endif  // CAFFE_COS_LAYER_HPP_
