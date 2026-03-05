"""Helios scheduler for MLX — DMD flow-matching with 3-stage pyramid support."""

import math

import mlx.core as mx
import numpy as np


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Compute dynamic shift (mu) based on spatial sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


class HeliosScheduler:
    """Flow-matching scheduler with shifted sigmas, DMD steps, and multi-stage
    pyramid support.

    For the Distilled model, each pyramid stage uses DMD (Distribution Matching
    Distillation) steps: predict x0 from flow, then re-noise with the original
    noisy tensor for all but the last step.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        stages: int = 3,
        stage_range: list | None = None,
        gamma: float = 1 / 3,
        use_dynamic_shifting: bool = True,
        base_image_seq_len: int = 256,
        max_image_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.stages = stages
        self.stage_range = stage_range or [0, 1 / 3, 2 / 3, 1]
        self.gamma = gamma
        self.use_dynamic_shifting = use_dynamic_shifting
        self.base_image_seq_len = base_image_seq_len
        self.max_image_seq_len = max_image_seq_len
        self.base_shift = base_shift
        self.max_shift = max_shift

        # Precompute global and per-stage schedules
        self.timestep_ratios = {}
        self.timesteps_per_stage = {}
        self.sigmas_per_stage = {}
        self.start_sigmas = {}
        self.end_sigmas = {}
        self.ori_start_sigmas = {}

        self._init_sigmas()
        self._init_sigmas_per_stage()

        self.sigma_min = float(self.global_sigmas[-1])
        self.sigma_max = float(self.global_sigmas[0])

        # Runtime state (set by set_timesteps)
        self.sigmas = None
        self.timesteps = None
        self._step_index = 0

    def _init_sigmas(self):
        """Compute the global shifted sigma schedule."""
        n = self.num_train_timesteps
        alphas = np.linspace(1, 1 / n, n + 1)
        sigmas = 1.0 - alphas
        sigmas = np.flip(
            self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        )[:-1].copy()
        self.global_sigmas = sigmas
        self.global_timesteps = sigmas * n

    def _init_sigmas_per_stage(self):
        """Compute per-stage sigma schedules with gamma correction."""
        n = self.num_train_timesteps
        stage_distance = []

        for i_s in range(self.stages):
            start_idx = int(self.stage_range[i_s] * n)
            start_idx = max(start_idx, 0)
            end_idx = int(self.stage_range[i_s + 1] * n)
            end_idx = min(end_idx, n)

            start_sigma = float(self.global_sigmas[start_idx])
            end_sigma = float(self.global_sigmas[end_idx]) if end_idx < n else 0.0
            self.ori_start_sigmas[i_s] = start_sigma

            if i_s != 0:
                ori_sigma = 1 - start_sigma
                gamma = self.gamma
                corrected = (
                    1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                ) * ori_sigma
                start_sigma = 1 - corrected

            stage_distance.append(start_sigma - end_sigma)
            self.start_sigmas[i_s] = start_sigma
            self.end_sigmas[i_s] = end_sigma

        tot_distance = sum(stage_distance)
        for i_s in range(self.stages):
            if i_s == 0:
                start_ratio = 0.0
            else:
                start_ratio = sum(stage_distance[:i_s]) / tot_distance
            if i_s == self.stages - 1:
                end_ratio = 1.0 - 1e-16
            else:
                end_ratio = sum(stage_distance[: i_s + 1]) / tot_distance

            self.timestep_ratios[i_s] = (start_ratio, end_ratio)

        for i_s in range(self.stages):
            ratio = self.timestep_ratios[i_s]
            t_max = min(float(self.global_timesteps[int(ratio[0] * n)]), 999)
            t_min = float(
                self.global_timesteps[min(int(ratio[1] * n), n - 1)]
            )
            self.timesteps_per_stage[i_s] = np.linspace(t_max, t_min, n + 1)[:-1]
            self.sigmas_per_stage[i_s] = np.linspace(0.999, 0, n + 1)[:-1]

    @staticmethod
    def _time_shift(mu: float, sigma: float, t):
        """Apply dynamic time shift: mu*t / (mu + (1-mu)*t)."""
        return mu * t / (mu + (1 - mu) * t)

    def set_timesteps(
        self,
        num_inference_steps: int,
        stage_index: int,
        image_seq_len: int | None = None,
        is_amplify_first_chunk: bool = False,
    ):
        """Set timesteps and sigmas for a specific pyramid stage with DMD.

        For DMD, num_inference_steps is expanded (+1, or *2+1 for amplify),
        then trimmed so that the final result has exactly num_inference_steps
        forward passes per stage.
        """
        # DMD expansion
        if is_amplify_first_chunk:
            n_steps = num_inference_steps * 2 + 1
        else:
            n_steps = num_inference_steps + 1

        n = self.num_train_timesteps
        stage_ts = self.timesteps_per_stage[stage_index]
        t_max, t_min = float(stage_ts[0]), float(stage_ts[-1])
        timesteps = np.linspace(t_max, t_min, n_steps)

        stage_sigmas = self.sigmas_per_stage[stage_index]
        s_max, s_min = float(stage_sigmas[0]), float(stage_sigmas[-1])
        sigmas = np.linspace(s_max, s_min, n_steps)
        sigmas = np.append(sigmas, 0.0)

        # DMD trim: drop last timestep, keep [sigmas[:-2], sigmas[-1:]]
        timesteps = timesteps[:-1]
        sigmas = np.concatenate([sigmas[:-2], sigmas[-1:]])

        # Dynamic shifting based on spatial resolution
        if self.use_dynamic_shifting and image_seq_len is not None:
            mu = calculate_shift(
                image_seq_len,
                self.base_image_seq_len,
                self.max_image_seq_len,
                self.base_shift,
                self.max_shift,
            )
            sigmas = self._time_shift(mu, 1.0, sigmas)
            # Remap timesteps to match shifted sigmas
            timesteps = (
                self.timesteps_per_stage[stage_index].min()
                + sigmas[:-1]
                * (
                    self.timesteps_per_stage[stage_index].max()
                    - self.timesteps_per_stage[stage_index].min()
                )
            )

        self.timesteps = mx.array(timesteps, dtype=mx.float32)
        self.sigmas = mx.array(sigmas, dtype=mx.float32)
        self._step_index = 0

    def step_dmd(
        self,
        model_output: mx.array,
        sample: mx.array,
        cur_step: int,
        noisy_start: mx.array,
    ) -> mx.array:
        """DMD step: predict x0 from flow, optionally re-noise.

        Args:
            model_output: Flow prediction from model
            sample: Current noisy latent (x_t)
            cur_step: Current step index within this stage
            noisy_start: Original noisy tensor for this stage (for re-noising)

        Returns:
            Denoised or re-noised sample
        """
        # Convert flow prediction to x0: x0 = x_t - sigma_t * flow
        sigma_t = self.sigmas[cur_step]
        x0_pred = sample - sigma_t * model_output

        num_timesteps = len(self.timesteps)
        if cur_step < num_timesteps - 1:
            # Re-noise: blend x0_pred with original noisy tensor at next sigma
            sigma_next = self.sigmas[cur_step + 1]
            prev_sample = (1 - sigma_next) * x0_pred + sigma_next * noisy_start
        else:
            prev_sample = x0_pred

        self._step_index = cur_step + 1
        return prev_sample

    def step(
        self,
        model_output: mx.array,
        sample: mx.array,
        sigma: mx.array | None = None,
        sigma_next: mx.array | None = None,
    ) -> mx.array:
        """Euler step: x_{t-1} = x_t + (sigma_next - sigma) * v."""
        if sigma is None:
            sigma = self.sigmas[self._step_index]
        if sigma_next is None:
            sigma_next = self.sigmas[self._step_index + 1]

        prev_sample = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return prev_sample

    def add_noise(
        self,
        original: mx.array,
        noise: mx.array,
        sigma: mx.array,
    ) -> mx.array:
        """Add noise at a given sigma level for flow matching."""
        return (1 - sigma) * original + sigma * noise
