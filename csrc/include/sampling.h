#pragma once
#include <torch/extension.h>

namespace aiter {
void top_p_sampling_from_probs(torch::Tensor probs, torch::Tensor output,
                               Optional<torch::Tensor> maybe_indices,
                               Optional<torch::Tensor> maybe_top_p_arr, double top_p_val,
                               bool deterministic, uint64_t philox_seed, uint64_t philox_offset);
}