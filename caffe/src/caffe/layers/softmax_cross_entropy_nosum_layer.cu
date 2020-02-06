#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_cross_entropy_nosum_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// NOTE: Implements softmax cross-entropy loss elementwise

template <typename Dtype>
__global__ void SoftmaxCrossEntropyNosumForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, dim * nthreads) {
    loss[index] = 0;
  }
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    loss[index * dim + label_value] = -log(max(prob_data[index * dim + label_value],
                    Dtype(FLT_MIN)));
    counts[index] = 1;
  }
}

template <typename Dtype>
__global__ void SoftmaxCrossEntropyNosumForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(i, n) {
    out[i] = in[i];
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyNosumLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxCrossEntropyNosumForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(outer_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_, prob_data, label, loss_data,
      outer_num_, dim, has_ignore_label_, ignore_label_, counts);
  SoftmaxCrossEntropyNosumForward<Dtype><<<CAFFE_GET_BLOCKS(outer_num_),
      CAFFE_CUDA_NUM_THREADS>>>(outer_num_ * dim, loss_data, top[0]->mutable_gpu_data());
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxCrossEntropyNosumBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int label_value = static_cast<int>(label[index]);
    bottom_diff[index * dim + label_value] -= 1;
    counts[index] = 1;
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyNosumLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxCrossEntropyNosumBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(outer_num_),
        CAFFE_CUDA_NUM_THREADS>>>(outer_num_, top_data, label, bottom_diff,
        outer_num_, dim, has_ignore_label_, ignore_label_, counts);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxCrossEntropyNosumLayer);

}  // namespace caffe