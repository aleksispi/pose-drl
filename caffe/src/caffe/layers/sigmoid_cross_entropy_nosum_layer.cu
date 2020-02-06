#include <vector>

#include "caffe/layers/sigmoid_cross_entropy_nosum_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// NOTE: IMPLEMENTS y * logy^ + (1-y) * log(1-y^) elementwise

template <typename Dtype>
__global__ void SigmoidCrossEntropyNosumForwardGPU(const int nthreads,
          const Dtype* input_data, const Dtype* target, Dtype* loss,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(i, nthreads) {
    const int target_value = static_cast<int>(target[i]);
    if (has_ignore_label_ && target_value == ignore_label_) {
      loss[i] = 0;
      counts[i] = 0;
    } else {
      loss[i] = -(input_data[i] * (target[i] - (input_data[i] >= 0)) -
          log(1 + exp(input_data[i] - 2 * input_data[i] *
          (input_data[i] >= 0))));
      counts[i] = 1;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyNosumIgnoreDiffGPU(const int count,
    const int ignore_label, const Dtype* target, Dtype* diff) {
  CUDA_KERNEL_LOOP(i, count) {
    const int target_value = static_cast<int>(target[i]);
    if (target_value == ignore_label) {
      diff[i] = 0;
    }
  }
}

template <typename Dtype>
__global__ void SigmoidCrossEntropyNosumForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = in[index];
  }
}

template <typename Dtype>
void SigmoidCrossEntropyNosumLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->gpu_data();
  const Dtype* target = bottom[1]->gpu_data();
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  Dtype* count_data = bottom[1]->mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidCrossEntropyNosumForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
      CAFFE_CUDA_NUM_THREADS>>>(count, input_data, target, loss_data,
      has_ignore_label_, ignore_label_, count_data);
  SigmoidCrossEntropyNosumForward<Dtype><<<CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS>>>(count, loss_data, top[0]->mutable_gpu_data());
}

template <typename Dtype>
void SigmoidCrossEntropyNosumLayer<Dtype>::Backward_gpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->gpu_data();
    const Dtype* target = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(count, sigmoid_output_data, bottom_diff);
    caffe_gpu_axpy(count, Dtype(-1), target, bottom_diff);
    // Zero out gradient of ignored targets.
    if (has_ignore_label_) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      SigmoidCrossEntropyNosumIgnoreDiffGPU<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>>(count, ignore_label_, target, bottom_diff);
    }
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SigmoidCrossEntropyNosumLayer);

}  // namespace caffe