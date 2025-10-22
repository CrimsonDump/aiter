
void top_p_sampling_from_probs(at::Tensor probs, at::Tensor uniform_samples, at::Tensor samples,
                               at::Tensor success, std::optional<at::Tensor> maybe_top_p_arr,
                               double top_p_val, bool deterministic, int64_t cuda_stream);