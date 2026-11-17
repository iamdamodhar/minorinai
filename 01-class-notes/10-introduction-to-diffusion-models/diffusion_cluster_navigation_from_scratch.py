"""
Introduction to Diffusion Models (DDPM & Text-Conditioned Cluster Navigation) — Pure Python
===========================================================================================
Zero-dependency standard-library implementation mirroring `GenAI_Fundamentals_Introduction_to_diffusers.ipynb`
and demonstrating the complete mathematical physics of:

  1. Forward Diffusion Closed-Form Jump (`DDPMScheduler.add_noise`):
       x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
  2. High-Probability Distribution Clusters & Exact Score Field:
       grad_{x_t} ln q(x_t) = - epsilon_theta(x_t, t) / sqrt(1 - alpha_bar_t)
  3. Reverse Noise Tracing (`DDPMScheduler.step`):
       Step-by-step trajectory from pure Gaussian static x_T -> final clean image x_0
  4. Text-to-Embedding Cross-Attention Conditioning & Classifier-Free Guidance (CFG)
  5. Multi-Word Compositional Mixing of High-Probability Clusters:
       Synthesizing a BRAND-NEW concept ("golden lotus" at [+0.80, -0.80]) by mixing the
       "golden" attribute cluster ([+0.80, +0.80]) and the "lotus" structure cluster ([-0.80, -0.80])!
"""

import math
import random


# ============================================================================
# 1. DDPM NOISE SCHEDULER (`DDPMScheduler` FROM NOTEBOOK CELLS 22-28 & 60)
# ============================================================================

class PurePythonDDPMScheduler:
    """Mirrors `diffusers.DDPMScheduler` (Linear & Cosine schedules)."""

    def __init__(self, num_train_timesteps: int = 1000, beta_start: float = 0.0001, beta_end: float = 0.02):
        self.num_train_timesteps = num_train_timesteps
        self.betas = [
            beta_start + (beta_end - beta_start) * (t / (num_train_timesteps - 1))
            for t in range(num_train_timesteps)
        ]
        self.alphas = [1.0 - b for b in self.betas]
        self.alphas_cumprod = []
        cum = 1.0
        for a in self.alphas:
            cum *= a
            self.alphas_cumprod.append(cum)

    def add_noise(self, x_0: list[float], noise: list[float], t: int) -> list[float]:
        """Closed-form forward jump: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise."""
        ab = self.alphas_cumprod[t]
        sqrt_ab = math.sqrt(ab)
        sqrt_one_minus_ab = math.sqrt(1.0 - ab)
        return [sqrt_ab * x_0[d] + sqrt_one_minus_ab * noise[d] for d in range(len(x_0))]

    def step(self, model_pred_noise: list[float], t: int, x_t: list[float], z: list[float] | None = None) -> list[float]:
        """
        Reverse DDPM Markov step (Cell 60 `noise_scheduler.step(residual, t, sample).prev_sample`):
          x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - ((1 - alpha_t) / sqrt(1 - alpha_bar_t)) * eps_theta) + sigma_t * z
        """
        alpha_t = self.alphas[t]
        beta_t = self.betas[t]
        ab_t = self.alphas_cumprod[t]
        coeff = (1.0 - alpha_t) / math.sqrt(1.0 - ab_t)
        mean = [(1.0 / math.sqrt(alpha_t)) * (x_t[d] - coeff * model_pred_noise[d]) for d in range(len(x_t))]
        if t == 0 or z is None:
            return mean
        sigma_t = math.sqrt(beta_t)
        return [mean[d] + sigma_t * z[d] for d in range(len(x_t))]


# ============================================================================
# 2. HIGH-PROBABILITY CLUSTERS, SCORE FIELD & MULTI-WORD CROSS-ATTENTION MIXING
# ============================================================================

# 3 High-Probability Training Clusters in 2D Semantic Space:
#   Dimension 0: Color Axis     (-0.80 = Royal Blue,   +0.80 = Golden Amber)
#   Dimension 1: Structure Axis (+0.80 = Butterfly,    -0.80 = Lotus Flower)
TRAINING_CLUSTERS = {
    "Cluster_A (Blue Butterfly)":   [-0.80, +0.80],
    "Cluster_B (Golden Butterfly)": [+0.80, +0.80],
    "Cluster_C (Blue Lotus)":       [-0.80, -0.80],
    # Notice: NO training image exists at [+0.80, -0.80] ("Golden Lotus")!
}

# Text Vocabulary Embeddings -> [Color-Value, Structure-Value, Color-Attn-Gate, Structure-Attn-Gate]
TOKEN_EMBEDDINGS = {
    "blue":      [-0.80,  0.00,  1.0, 0.0],  # Binds strongly to Dimension 0 (Color)
    "golden":    [+0.80,  0.00,  1.0, 0.0],  # Binds strongly to Dimension 0 (Color)
    "butterfly": [ 0.00, +0.80,  0.0, 1.0],  # Binds strongly to Dimension 1 (Structure)
    "lotus":     [ 0.00, -0.80,  0.0, 1.0],  # Binds strongly to Dimension 1 (Structure)
}


def exact_marginal_score_and_noise(
    x_t: list[float],
    t: int,
    scheduler: PurePythonDDPMScheduler,
    centers: list[list[float]],
    cluster_sigma: float = 0.06,
) -> tuple[list[float], list[float], list[float]]:
    """
    Computes the exact analytical Score grad_{x_t} ln q_t(x_t) and optimal DDPM predicted noise
    eps*(x_t, t) = -sqrt(1 - alpha_bar_t) * grad_{x_t} ln q_t(x_t) over a Gaussian Mixture of clusters!
    """
    ab_t = scheduler.alphas_cumprod[t]
    sqrt_ab = math.sqrt(ab_t)
    var_t = ab_t * (cluster_sigma ** 2) + (1.0 - ab_t)

    # Compute log-probabilities for each high-probability cluster k
    log_weights = []
    for mu in centers:
        scaled_mu = [sqrt_ab * m for m in mu]
        dist_sq = sum((x_t[d] - scaled_mu[d]) ** 2 for d in range(len(x_t)))
        log_weights.append(-0.5 * dist_sq / var_t)

    max_lw = max(log_weights)
    exps = [math.exp(lw - max_lw) for lw in log_weights]
    sum_exps = sum(exps)
    responsibilities = [e / sum_exps for e in exps]  # Posterior P(Cluster k | x_t)

    # Expected clean target E[x_0 | x_t] (Tweedie's Denoised Estimate)
    x0_hat = [
        sum(responsibilities[k] * centers[k][d] for k in range(len(centers)))
        for d in range(len(x_t))
    ]
    # Score grad_{x_t} ln q_t(x_t) = -(x_t - sqrt(ab_t) * E[x_0 | x_t]) / (1 - ab_t)
    sqrt_one_minus_ab = math.sqrt(1.0 - ab_t)
    pred_noise = [
        (x_t[d] - sqrt_ab * x0_hat[d]) / max(sqrt_one_minus_ab, 1e-8)
        for d in range(len(x_t))
    ]
    score = [-pred_noise[d] / max(sqrt_one_minus_ab, 1e-8) for d in range(len(x_t))]
    return pred_noise, score, responsibilities


def encode_prompt_cross_attention(prompt: str) -> tuple[list[float], list[float]]:
    """
    Converts a multi-word text prompt (e.g., 'golden lotus') into token embeddings and
    computes Cross-Attention attribute routing weights across [Color_Dim, Structure_Dim].
    """
    words = prompt.lower().split()
    embs = [TOKEN_EMBEDDINGS[w] for w in words if w in TOKEN_EMBEDDINGS]
    # Cross-Attention routes Color words to Dim 0 and Structure words to Dim 1
    color_weights = [e[2] for e in embs]
    struct_weights = [e[3] for e in embs]

    c_sum = sum(color_weights) or 1.0
    s_sum = sum(struct_weights) or 1.0

    target_color = sum(e[0] * e[2] for e in embs) / c_sum
    target_struct = sum(e[1] * e[3] for e in embs) / s_sum
    return [target_color, target_struct], [c_sum, s_sum]


def conditional_cfg_noise_prediction(
    x_t: list[float],
    t: int,
    scheduler: PurePythonDDPMScheduler,
    prompt: str,
    cfg_scale: float = 3.0,
) -> list[float]:
    """
    Implements Compositional Score Mixing + Classifier-Free Guidance (CFG):
      eps_CFG = eps_uncond + w_CFG * (eps_cond(prompt) - eps_uncond)
    where each word in `prompt` pulls its attended attribute dimension toward the
    matching high-probability cluster manifold!
    """
    centers = list(TRAINING_CLUSTERS.values())
    eps_uncond, _, _ = exact_marginal_score_and_noise(x_t, t, scheduler, centers)

    ab_t = scheduler.alphas_cumprod[t]
    sqrt_ab = math.sqrt(ab_t)
    sqrt_one_minus_ab = math.sqrt(1.0 - ab_t)

    # Multi-word Cross-Attention composes attribute pulls from the training clusters:
    # - Color words ("golden") query clusters matching color -> Cluster_B (+0.80)
    # - Structure words ("lotus") query clusters matching structure -> Cluster_C (-0.80)
    composed_x0_target, active_gates = encode_prompt_cross_attention(prompt)

    eps_cond = []
    for d in range(2):
        if active_gates[d] > 0:
            eps_d = (x_t[d] - sqrt_ab * composed_x0_target[d]) / max(sqrt_one_minus_ab, 1e-8)
        else:
            eps_d = eps_uncond[d]
        eps_cond.append(eps_d)

    # Classifier-Free Guidance (CFG) vector extrapolation
    return [
        eps_uncond[d] + cfg_scale * (eps_cond[d] - eps_uncond[d])
        for d in range(2)
    ]


# ============================================================================
# 3. RUN COMPLETE STEP-BY-STEP DIFFUSION EXPERIMENTS
# ============================================================================

def run_diffusion_experiments() -> None:
    print("=" * 94)
    print(" 1. FORWARD DIFFUSION SCHEDULE (`DDPMScheduler`: Signal sqrt(alpha_bar_t) vs Noise)")
    print("=" * 94)
    sched_1000 = PurePythonDDPMScheduler(num_train_timesteps=1000)
    clean_butterfly = [-0.80, +0.80]  # Cluster A: Blue Butterfly
    fixed_noise = [+0.50, -1.20]

    for t in [0, 100, 250, 500, 750, 999]:
        ab = sched_1000.alphas_cumprod[t]
        x_t = sched_1000.add_noise(clean_butterfly, fixed_noise, t)
        print(
            f"  Timestep t={t:3d} | alpha_bar_t={ab:.4f} | Signal Scale={math.sqrt(ab):.4f} "
            f"| Noise Scale={math.sqrt(1-ab):.4f} | Noisy x_t=[{x_t[0]:+6.3f}, {x_t[1]:+6.3f}]"
        )

    print("\n" + "=" * 94)
    print(" 2. UNCONDITIONAL REVERSE NAVIGATION: NOISE TRACING TO HIGH-PROBABILITY CLUSTERS")
    print("    Training Clusters: A=[-0.80, +0.80] (Blue Butterfly), B=[+0.80, +0.80] (Golden Butterfly),")
    print("                       C=[-0.80, -0.80] (Blue Lotus)")
    print("=" * 94)
    sched = PurePythonDDPMScheduler(num_train_timesteps=50, beta_start=0.001, beta_end=0.12)
    centers = list(TRAINING_CLUSTERS.values())
    rng = random.Random(10)

    # Start from pure noise x_T
    x = [-0.35, +0.45]
    for t in reversed(range(50)):
        eps_pred, score, resp = exact_marginal_score_and_noise(x, t, sched, centers)
        z = [rng.gauss(0.0, 0.15) for _ in range(2)] if t > 0 else None
        x = sched.step(eps_pred, t, x, z)
        if t in (49, 35, 20, 10, 0):
            print(
                f"  Reverse Step t={t:2d} | x_t=[{x[0]:+6.3f}, {x[1]:+6.3f}] "
                f"| Uphill Score=[{score[0]:+6.2f}, {score[1]:+6.2f}] "
                f"| Cluster Prob P(A,B,C)=[{resp[0]:.2f}, {resp[1]:.2f}, {resp[2]:.2f}]"
            )
    print(f"  --> Landed cleanly inside High-Probability Cluster A (Blue Butterfly [-0.80, +0.80]): [{x[0]:+.3f}, {x[1]:+.3f}]!")

    print("\n" + "=" * 94)
    print(" 3. MULTI-WORD COMPOSITIONAL CLUSTER MIXING: CREATING A BRAND-NEW 'GOLDEN LOTUS' IMAGE!")
    print("    Prompt: 'golden lotus' -> Mixes 'golden' from Cluster B [+0.80, *] + 'lotus' from Cluster C [*, -0.80]")
    print("=" * 94)
    x_new = [-0.20, +0.30]  # Start far away in the wrong quadrant!
    for t in reversed(range(50)):
        eps_cfg = conditional_cfg_noise_prediction(x_new, t, sched, prompt="golden lotus", cfg_scale=1.0)
        x_new = sched.step(eps_cfg, t, x_new, z=None)
        if t in (49, 35, 20, 10, 0):
            print(f"  Reverse Step t={t:2d} | Prompt='golden lotus' | Trajectory x_t = [{x_new[0]:+6.3f}, {x_new[1]:+6.3f}]")
    print(f"  --> Synthesized Brand-New 'Golden Lotus' Image x_0 = [{x_new[0]:+.3f}, {x_new[1]:+.3f}] (Target: [+0.800, -0.800])!")

    print("\n" + "=" * 94)
    print(" 4. HOW HIGH-DIMENSIONAL LATENT/FEATURE SPACE CONVERTS TO RGB PIXEL SPACE (`show_images`)")
    print("=" * 94)
    # Suppose our synthesized 'Golden Lotus' vector x_new = [+0.800, -0.800] sits in a 4-channel
    # bottleneck [Color_Hue, Structure_Type, Wing_or_Petal_Symmetry=+0.90, Background_Green=-0.40]:
    high_dim_feature = [x_new[0], x_new[1], +0.90, -0.40]

    # Final conv_out (3x4 projection matrix) converts 4 semantic channels -> 3 RGB channels in [-1, +1]
    W_rgb = [
        [+0.85, -0.15, +0.10, -0.05],  # Red channel filter
        [+0.45, -0.10, +0.15, +0.20],  # Green channel filter
        [-0.80, +0.20, -0.10, -0.15],  # Blue channel filter
    ]
    rgb_float_minus1_to_1 = [
        max(-1.0, min(1.0, sum(W_rgb[c][k] * high_dim_feature[k] for k in range(4))))
        for c in range(3)
    ]
    # Exact `show_images(x)` conversion from Cell 13: x_01 = x * 0.5 + 0.5 -> uint8 = int(x_01 * 255)
    rgb_0_to_1 = [v * 0.5 + 0.5 for v in rgb_float_minus1_to_1]
    rgb_uint8 = [int(round(v * 255)) for v in rgb_0_to_1]

    print(f"  Stage A (High-Dim Semantic Vector in R^4)      : {[round(v, 3) for v in high_dim_feature]}")
    print(f"  Stage B (Projected RGB Tensor in [-1, +1]^3)   : [R={rgb_float_minus1_to_1[0]:+.3f}, G={rgb_float_minus1_to_1[1]:+.3f}, B={rgb_float_minus1_to_1[2]:+.3f}]")
    print(f"  Stage C (Shifted to [0, 1] via x*0.5 + 0.5)    : [R={rgb_0_to_1[0]:.3f},  G={rgb_0_to_1[1]:.3f},  B={rgb_0_to_1[2]:.3f}]")
    print(f"  Stage D (Screen Pixel uint8 [0..255] Triplet)  : RGB({rgb_uint8[0]}, {rgb_uint8[1]}, {rgb_uint8[2]}) --> Warm Golden-Amber Pixel!")

    print("\n" + "=" * 94)
    print(" 5. MATHEMATICAL PROOF: GAUSSIAN KL-DIVERGENCE D_KL(q || p_theta) === SCALED F.mse_loss(eps)")
    print("=" * 94)
    # Discrete Goa Weather KL-Divergence verification
    P_goa = [0.90, 0.10]
    Q_friend = [0.20, 0.80]
    kl_discrete = sum(p * math.log(p / q) for p, q in zip(P_goa, Q_friend))
    print(f"  [Discrete KL] P={P_goa} vs Q={Q_friend} --> D_KL(P || Q) = {kl_discrete:.4f} nats (D_KL(P || P) = 0.0000)")

    # DDPM Step-t Gaussian KL-Divergence vs. Noise MSE equivalence at t = 250
    t_test = 250
    alpha_t = sched_1000.alphas[t_test]
    beta_t = sched_1000.betas[t_test]
    ab_t = sched_1000.alphas_cumprod[t_test]
    sigma_t_sq = beta_t

    x_0_sample = [-0.80, +0.80]
    true_eps = [+0.50, -1.20]
    pred_eps = [+0.35, -0.95]  # UNet's predicted noise eps_theta(x_t, t)
    x_t_sample = sched_1000.add_noise(x_0_sample, true_eps, t_test)

    # 1. Compute Teacher Mean mu_tilde(x_t, eps) and Student Mean mu_theta(x_t, eps_theta)
    c_eps = beta_t / math.sqrt(1.0 - ab_t)
    mu_tilde = [(1.0 / math.sqrt(alpha_t)) * (x_t_sample[d] - c_eps * true_eps[d]) for d in range(2)]
    mu_theta = [(1.0 / math.sqrt(alpha_t)) * (x_t_sample[d] - c_eps * pred_eps[d]) for d in range(2)]

    # Exact Gaussian KL: D_KL(N(mu_tilde, sigma_t^2 I) || N(mu_theta, sigma_t^2 I))
    kl_gaussian = sum((mu_tilde[d] - mu_theta[d]) ** 2 for d in range(2)) / (2.0 * sigma_t_sq)

    # 2. Compute Scaled Noise Squared Error: w_t * ||eps - eps_theta||_2^2
    w_t = (beta_t ** 2) / (2.0 * sigma_t_sq * alpha_t * (1.0 - ab_t))
    noise_sq_err = sum((true_eps[d] - pred_eps[d]) ** 2 for d in range(2))
    mse_loss_val = noise_sq_err / 2.0  # PyTorch F.mse_loss averages over D=2 dimensions

    print(f"  [DDPM Step t={t_test}] Exact Gaussian D_KL(q || p_theta) = {kl_gaussian:.10f}")
    print(f"  [DDPM Step t={t_test}] w_t * ||eps - eps_theta||_2^2     = {w_t * noise_sq_err:.10f}  (Identical to 12 decimals!)")
    print(f"  [Cell 40 L_simple]   F.mse_loss(noise_pred, noise)     = {mse_loss_val:.6f}      (Unweighted MSE)")

    print("\n" + "=" * 94)
    print(" 6. LEARNING RATE SCHEDULES IN DIFFUSION (Peak lr=1e-4, Warmup=200 steps, Total=1000 steps)")
    print("=" * 94)
    lr_max = 1e-4
    t_warm = 200
    t_tot = 1000
    print(f"  {'Step':>5s} | {'Constant':>10s} | {'Warm+Const':>10s} | {'Warm+Linear':>11s} | {'Warm+Cosine':>11s} | {'StepLR(0.3x)':>12s}")
    print("  " + "-" * 74)
    for step in [0, 50, 100, 200, 400, 500, 600, 800, 950, 1000]:
        lr_const = lr_max
        lr_warm_const = lr_max * (step / t_warm) if step < t_warm else lr_max
        if step < t_warm:
            lr_linear = lr_max * (step / t_warm)
            lr_cosine = lr_max * (step / t_warm)
        else:
            prog = (step - t_warm) / (t_tot - t_warm)
            lr_linear = lr_max * (1.0 - prog)
            lr_cosine = 0.5 * lr_max * (1.0 + math.cos(math.pi * prog))
        lr_step = lr_max * (0.3 ** (step // 400))
        print(
            f"  {step:5d} | {lr_const:10.2e} | {lr_warm_const:10.2e} | "
            f"{lr_linear:11.2e} | {lr_cosine:11.2e} | {lr_step:12.2e}"
        )
    print("=" * 94)


if __name__ == "__main__":
    run_diffusion_experiments()
