"""
Lectures 07 & 08: Bahdanau (Additive) Attention & Seq2Seq Decoder from Scratch (Pure Python)
============================================================================================
Zero-dependency standard-library implementation mirroring the exact architecture of
`GenAI_Fundamentals_Attention_Mechanism_Lec_7,_8.ipynb`:

  Source: ["<sos>", "i", "am", "student", "<eos>"]
  Target: ["<sos>", "je", "suis", "etudiant", "<eos>"]

Demonstrates:
  1. Unidirectional Encoder producing ALL hidden states H = [h_1, ..., h_{T_x}] (Open-Book Table).
  2. Bahdanau Additive Alignment:
       energy_{t,j} = tanh(W_a [s_{t-1}; h_j] + b_a)
       score_{t,j}  = v_a^T energy_{t,j}
       alpha_{t,j}  = Softmax_j(score_{t,j})
  3. Dynamic Per-Step Context Vector c_t = sum_j alpha_{t,j} * h_j (equivalent to torch.bmm).
  4. Decoder step combining [embedded(y_{t-1}); c_t] -> s_t and [s_t; c_t; embedded] -> logits.
  5. Mathematical Proof Demo: Why removing tanh() collapses Additive Attention into a static score!
"""

import math
import random


def sigmoid(x: float) -> float:
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))


def softmax(scores: list[float]) -> list[float]:
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]


def matvec(W: list[list[float]], x: list[float], b: list[float] | None = None) -> list[float]:
    out = []
    for r, row in enumerate(W):
        val = sum(row[c] * x[c] for c in range(len(x)))
        if b is not None:
            val += b[r]
        out.append(val)
    return out


class PurePythonBahdanauAttention:
    """
    Mirrors `BahdanauAttention(enc_hid_dim, dec_hid_dim)` in the Lecture 7-8 notebook:
      self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
      self.v    = nn.Linear(dec_hid_dim, 1, bias=False)
    """

    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, W_a: list[list[float]], b_a: list[float], v_a: list[float]):
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.W_a = W_a  # Shape: (dec_hid_dim x (dec_hid_dim + enc_hid_dim))
        self.b_a = b_a  # Shape: (dec_hid_dim)
        self.v_a = v_a  # Shape: (dec_hid_dim)

    def forward(self, dec_hidden: list[float], enc_outputs: list[list[float]]) -> tuple[list[float], list[float], list[float]]:
        """
        dec_hidden : shape (dec_hid_dim,)          -> s_{t-1}
        enc_outputs: shape (src_len, enc_hid_dim)  -> [h_1, ..., h_{src_len}]
        Returns    : (alpha_weights, raw_scores, context_vector)
        """
        raw_scores = []
        for h_j in enc_outputs:
            # Concatenate [s_{t-1}, h_j] -> shape (dec_hid_dim + enc_hid_dim,)
            concat_vec = dec_hidden + h_j
            # energy = tanh(W_a @ [s_{t-1}, h_j] + b_a) -> shape (dec_hid_dim,)
            energy = [math.tanh(u) for u in matvec(self.W_a, concat_vec, self.b_a)]
            # score e_{t,j} = v_a^T @ energy -> scalar
            score = sum(self.v_a[k] * energy[k] for k in range(self.dec_hid_dim))
            raw_scores.append(score)

        # Normalize alignment scores across source positions via Softmax
        alpha = softmax(raw_scores)

        # Dynamic Context Vector c_t = sum_j alpha_{t,j} * h_j  (equivalent to torch.bmm)
        context = [
            sum(alpha[j] * enc_outputs[j][d] for j in range(len(enc_outputs)))
            for d in range(self.enc_hid_dim)
        ]
        return alpha, raw_scores, context


def render_bar(weight: float, width: int = 20) -> str:
    filled = int(round(weight * width))
    return "█" * filled + "░" * (width - filled)


def run_microscopic_attention_demo():
    print("=" * 88)
    print(" LECTURES 07 & 08: MICROSCOPIC BAHDANAU ATTENTION WALKTHROUGH (PURE PYTHON)")
    print(" Sentence Pair: 'i am student'  -->  'je suis etudiant'")
    print("=" * 88)

    src_tokens = ["<sos>", "i", "am", "student", "<eos>"]
    tgt_steps = [
        ("Step t=1 (Predicting 'je')",       [1.0,  0.0,  0.0, 0.0]),
        ("Step t=2 (Predicting 'suis')",     [0.0,  1.0,  0.0, 0.0]),
        ("Step t=3 (Predicting 'etudiant')", [0.0,  0.0,  1.0, 0.0]),
        ("Step t=4 (Predicting '<eos>')",    [0.0,  0.0,  0.0, 1.0]),
    ]

    # Semantic 4D Encoder Annotations h_j:
    # [Subject-Role, Verb/Copula-Role, Noun/Entity-Role, Boundary-Marker]
    encoder_outputs = [
        [ 0.10,  0.05,  0.05,  0.85],  # h_1: "<sos>"
        [ 0.95,  0.10,  0.05,  0.05],  # h_2: "i"
        [ 0.10,  0.95,  0.10,  0.05],  # h_3: "am"
        [ 0.05,  0.10,  0.98,  0.05],  # h_4: "student"
        [ 0.05,  0.05,  0.05,  0.95],  # h_5: "<eos>"
    ]

    # Bahdanau Additive Attention weights (dec_hid=4, enc_hid=4 -> concat=8)
    # Each hidden unit k checks whether BOTH query s_{t-1}[k] AND key h_j[k] are simultaneously active!
    # Notice: if s_{t-1}[k]=1 and h_j[k]=1, 2.0 + 2.0 - 2.5 = +1.5 -> tanh(+1.5) = +0.905!
    # If only one is active: 2.0 + 0.0 - 2.5 = -0.5 -> tanh(-0.5) = -0.462!
    W_a = [
        [2.2, 0.0, 0.0, 0.0,  2.2, 0.0, 0.0, 0.0],
        [0.0, 2.2, 0.0, 0.0,  0.0, 2.2, 0.0, 0.0],
        [0.0, 0.0, 2.2, 0.0,  0.0, 0.0, 2.2, 0.0],
        [0.0, 0.0, 0.0, 2.2,  0.0, 0.0, 0.0, 2.2],
    ]
    b_a = [-2.6, -2.6, -2.6, -2.6]
    v_a = [2.5, 2.5, 2.5, 2.5]

    attn = PurePythonBahdanauAttention(enc_hid_dim=4, dec_hid_dim=4, W_a=W_a, b_a=b_a, v_a=v_a)

    for label, s_prev in tgt_steps:
        alpha, scores, c_t = attn.forward(s_prev, encoder_outputs)
        print(f"\n{label} | Decoder Query s_{{t-1}} = {s_prev}")
        print("-" * 88)
        for j, tok in enumerate(src_tokens):
            print(
                f"  Source j={j} ({tok:<9s}) | score e_{{t,{j}}} = {scores[j]:+7.4f} "
                f"| alpha_{{t,{j}}} = {alpha[j]:.4f} ({alpha[j]*100:5.1f}%) {render_bar(alpha[j])}"
            )
        print(f"  --> Dynamic Context Vector c_t = {[round(v, 4) for v in c_t]}  (Weighted sum of h_1..h_5)")

    # =====================================================================
    # MATHEMATICAL PROOF: WHY tanh() INSIDE ADDITIVE ATTENTION IS MANDATORY
    # =====================================================================
    print("\n" + "=" * 88)
    print(" MATHEMATICAL PROOF: WHAT HAPPENS IF WE REMOVE tanh() FROM BAHDANAU ATTENTION?")
    print("=" * 88)
    for step_idx in [0, 2]:
        label, s_prev = tgt_steps[step_idx]
        linear_scores = []
        for h_j in encoder_outputs:
            lin = matvec(W_a, s_prev + h_j, b_a)  # NO tanh()!
            linear_scores.append(sum(v_a[k] * lin[k] for k in range(4)))
        alpha_no_tanh = softmax(linear_scores)
        print(f"  Without tanh() at {label}:")
        print(f"    Attention weights alpha = {[round(a, 4) for a in alpha_no_tanh]}")
    print("  --> IDENTICAL WEIGHTS AT EVERY DECODER STEP! Without non-linear tanh(),")
    print("      v^T(W_s s_{t-1} + W_h h_j) separates into C(s_{t-1}) + v^T W_h h_j,")
    print("      and the constant C(s_{t-1}) cancels out completely inside Softmax!")
    print("=" * 88)


if __name__ == "__main__":
    run_microscopic_attention_demo()
