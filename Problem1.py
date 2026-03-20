#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import matplotlib.pyplot as plt

CONST_VALUE_MIN = 0
CONST_VALUE_MAX = 255


# Step 1: Noisy image Y
def noisy_image_Y(filepath):
    image = np.loadtxt(filepath)
    return image


# Step 2: Gibbs sampling for image denoising
def gibbs_sampling(
    noisy_obs, sigma=25, lambda_param=0.01, sweeps=500, burn_in=200, seed=0
):
    """
    Gaussian denoising:
      x_ij ∈ {0,1}
      y_ij ~ N(x_ij, sigma^2)
      prior prefers agreement with 4-neighbors
    """
    rng = np.random.default_rng(seed)
    H, W = noisy_obs.shape

    # init from observation so the square doesn't disappear
    x = noisy_obs.copy()

    # degree (# of valid 4-neighbors) per pixel (edges/corners smaller)
    deg = np.full((H, W), 4, dtype=np.float64)
    deg[0, :] -= 1
    deg[-1, :] -= 1
    deg[:, 0] -= 1
    deg[:, -1] -= 1

    acc = np.zeros((H, W), dtype=np.float64)
    acc_count = 0

    def neighbor_sum(curr):
        nsum = np.zeros_like(curr, dtype=np.float64)
        nsum[1:, :] += curr[:-1, :]  # up
        nsum[:-1, :] += curr[1:, :]  # down
        nsum[:, 1:] += curr[:, :-1]  # left
        nsum[:, :-1] += curr[:, 1:]  # right
        return nsum

    sigma2 = sigma**2

    for s in range(sweeps):
        # scanline order
        for i in range(H):
            for j in range(W):
                # compute conditional distribution p(x_ij | x_-ij, y)
                nsum = neighbor_sum(x)
                v = 1.0 / (1.0 / sigma2 + lambda_param * deg[i, j])
                mu = v * (noisy_obs[i, j] / sigma2 + lambda_param * nsum[i, j])

                # N(mu,v)
                x[i, j] = rng.normal(mu, np.sqrt(v))

        # average samples after burn-in (this is what makes it look “denoised” instead of noisy)
        if s >= burn_in:
            acc += x
            acc_count += 1

    # compute final denoised estimate by averageing 300 samples images pixelwise
    post_mean = acc / max(acc_count, 1)
    return post_mean


# --- run ---
if len(sys.argv) < 2:
    print("Usage: python problem1.py <input_file.txt>")
    sys.exit(1)

noisy_image_Y = noisy_image_Y(sys.argv[1])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the noisy image Y
axes[0].imshow(noisy_image_Y, cmap="gray", vmin=CONST_VALUE_MIN, vmax=CONST_VALUE_MAX)
axes[0].set_title("Noisy Input Y")
axes[0].axis("off")

# Count and display the denoised image X
denoised_image_X = gibbs_sampling(
    noisy_image_Y, sigma=25, lambda_param=0.01, sweeps=500, burn_in=200, seed=0
)
axes[1].imshow(
    denoised_image_X, cmap="gray", vmin=CONST_VALUE_MIN, vmax=CONST_VALUE_MAX
)
axes[1].set_title("Denoised Output X")
axes[1].axis("off")
plt.tight_layout()
plt.savefig("problem1_output.png", dpi=150, bbox_inches="tight")
print("Denoising complete. Output saved as 'problem1_output.png'.")
plt.show()
