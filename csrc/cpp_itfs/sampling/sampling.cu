
void top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               at::Tensor success, std::optional<at::Tensor> maybe_top_p_arr,
                               double top_p_val, bool deterministic, int64_t cuda_stream) {
  CHECK_INPUT(probs);
  CHECK_INPUT(uniform_samples);
  auto device = probs.device();
  CHECK_EQ(uniform_samples.device(), device);
  CHECK_DIM(2, probs);            // probs: (batch_size, vocab_size)
  CHECK_DIM(2, uniform_samples);  // uniform_samples: (max_top_p_rounds, batch_size)
  CHECK_EQ(probs.size(0), uniform_samples.size(1));
  unsigned int batch_size = probs.size(0);
  unsigned int vocab_size = probs.size(1);
  unsigned int max_top_p_rounds = uniform_samples.size(0);
  bool has_top_p_arr = maybe_top_p_arr.has_value();

  cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaError_t status = sampling::TopPSamplingFromProb<float, int>(
      static_cast<float*>(probs.data_ptr()), static_cast<float*>(uniform_samples.data_ptr()),
      static_cast<int*>(samples.data_ptr()), static_cast<bool*>(success.data_ptr()),
      has_top_p_arr ? static_cast<float*>(maybe_top_p_arr->data_ptr()) : nullptr, batch_size,
      top_p_val, vocab_size, max_top_p_rounds, deterministic, stream);
  TORCH_CHECK(status == cudaSuccess, "TopPSamplingFromProbs failed with error code " +
                                         std::string(cudaGetErrorString(status)));
}