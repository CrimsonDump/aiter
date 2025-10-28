#include "sampling.cuh"

namespace aiter {
void top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               Optional<torch::Tensor> maybe_indices,
                               Optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset) {
//   CHECK_INPUT(probs);
//   CHECK_DIM(2, probs);  // probs: (batch_size, vocab_size)
  unsigned int batch_size = output.sizes()[0];
  unsigned int vocab_size = probs.sizes()[1];
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  cudaSetDevice(probs->device.device_id);
//   auto stream = get_stream(probs->device);
  cudaError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<int*>(output.data_ptr()),
      maybe_indices.has_value() ? static_cast<int*>(maybe_indices.value().data_ptr()) : nullptr,
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr.value().data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, deterministic, philox_seed, philox_offset, /*stream*/0);
  TORCH_CHECK(status == cudaSuccess, "TopPSamplingFromProbs failed with error code " + cudaGetErrorString(status));
}
}