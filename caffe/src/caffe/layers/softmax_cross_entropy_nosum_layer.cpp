#include <algorithm>
#include <cfloat>
#include <vector>
#include <unistd.h>
#include <iostream>


#include "caffe/layers/softmax_cross_entropy_nosum_layer.hpp"
#include "caffe/util/math_functions.hpp"

using namespace std;

// NOTE: Implements softmax cross-entropy loss elementwise

namespace caffe {

template <typename Dtype>
void SoftmaxCrossEntropyNosumLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyNosumLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  CHECK_EQ(outer_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyNosumLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  // outer_num_ = batch_size
  // dim = number of classes / categories in softmax
  //
  // The below loop fills in the "dim-by-outer_num_ matrix"
  // such that every column has exactly one non-zero element,
  // being equal to the negative log-probability of the
  // correct category (label)
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < dim; j++) {
      const int label_value = static_cast<int>(label[i]);
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, dim);
      if (j == label_value) {
        top_data[i * dim + j] =
         -log(std::max(prob_data[i * dim + label_value], Dtype(FLT_MIN)));
      } else {
        top_data[i * dim + j] = 0;
      }
    }
    ++count;
  }
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyNosumLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    // Below line of code: Sets dL / dx = smax-part.
    // In particular, recall that the i:th derivative
    // of softmax is
    //
    // dL / dx_i = -1(i == label) + smax(i)
    //
    // where smax is nothing but prob_data.
    // Thus after below line of code, need to also
    // add the -1 at appropriate places...
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    // This for-loop adds that -1 appropriately
    for (int i = 0; i < outer_num_; ++i) {
      const int label_value = static_cast<int>(label[i]);
      bottom_diff[i * dim + label_value] -= 1;
      ++count;
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxCrossEntropyNosumLayer);
#endif

INSTANTIATE_CLASS(SoftmaxCrossEntropyNosumLayer);
REGISTER_LAYER_CLASS(SoftmaxCrossEntropyNosum);

}  // namespace caffe