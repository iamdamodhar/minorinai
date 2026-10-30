"""
Take-Home Assignment Verification Script: Handwritten LSTM for Next-Word Generation
===================================================================================
Problem Settings (Exact Match to Assignment PDF):
  - Vocabulary size (V)     = 5   ({0: "I", 1: "love", 2: "deep", 3: "learning", 4: "<EOS>"})
  - Hidden state size (h)   = 2
  - Token embedding size (e)= 2
  - Context size (c)        = 3   (Input Token IDs: [0, 1, 2] -> "I love deep")

Equations Used (Concatenated Form [h_{t-1}, x_t] of shape (4 x 1)):
  1. f_t      = sigmoid(W_f [h_{t-1}, x_t] + b_f)        Shape: (2x4)(4x1) + (2x1) = (2x1)
  2. i_t      = sigmoid(W_i [h_{t-1}, x_t] + b_i)        Shape: (2x4)(4x1) + (2x1) = (2x1)
  3. o_t      = sigmoid(W_o [h_{t-1}, x_t] + b_o)        Shape: (2x4)(4x1) + (2x1) = (2x1)
  4. C_tilde  = tanh(W_c [h_{t-1}, x_t] + b_c)           Shape: (2x4)(4x1) + (2x1) = (2x1)
  5. C_t      = f_t * C_{t-1} + i_t * C_tilde            Shape: (2x1)
  6. h_t      = o_t * tanh(C_t)                          Shape: (2x1)
  7. y        = W_out h_3 + b_out                        Shape: (5x2)(2x1) + (5x1) = (5x1)
  8. P        = Softmax(y)                               Shape: (5x1) -> argmax = 3 ("learning")
"""

import math
from decimal import Decimal, ROUND_HALF_UP


def r4(val: float) -> float:
    """Standard schoolbook half-up rounding to 4 decimal places (matches pocket calculator)."""
    return float(Decimal(str(val)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def sigmoid(u: float) -> float:
    return 1.0 / (1.0 + math.exp(-u))


# =====================================================================
# 1. HYPERPARAMETERS, VOCABULARY & EMBEDDING TABLE (V=5, e=2, h=2, c=3)
# =====================================================================
V = 5
H = 2
E_DIM = 2
C_LEN = 3

VOCAB = {0: "I", 1: "love", 2: "deep", 3: "learning", 4: "<EOS>"}

# Embedding Matrix E of shape (V x e) = (5 x 2)
E_MATRIX = [
    [1.0,  0.0],   # ID 0: "I"
    [0.5,  1.0],   # ID 1: "love"
    [0.0,  1.0],   # ID 2: "deep"
    [-0.5, 1.0],   # ID 3: "learning"
    [0.0, -1.0],   # ID 4: "<EOS>"
]

# =====================================================================
# 2. LSTM WEIGHTS (2 x 4) & BIASES (2 x 1) + LOGITS LAYER (5 x 2)
# =====================================================================
W_f = [
    [0.5,  0.0,  0.5,  0.2],
    [0.0,  0.5, -0.2,  0.5],
]
b_f = [0.5, 0.5]

W_i = [
    [0.4, -0.2,  0.8,  0.2],
    [-0.2, 0.4,  0.2,  0.8],
]
b_i = [0.0, 0.0]

W_o = [
    [0.5,  0.2,  0.6,  0.2],
    [0.2,  0.5,  0.2,  0.6],
]
b_o = [0.1, 0.1]

W_c = [
    [0.5,  0.0,  1.0, -0.5],
    [0.0,  0.5, -0.5,  1.0],
]
b_c = [0.0, 0.0]

W_out = [
    [0.5,  -1.0],
    [0.8,   0.2],
    [-0.4,  0.6],
    [-0.5,  2.0],
    [-0.2, -0.8],
]
b_out = [0.0, 0.0, 0.1, 0.5, -0.2]


def compute_gate_explicit(name: str, W: list[list[float]], z: list[float], b: list[float], act_name: str):
    """Computes W @ z + b and applies activation with explicit step-by-step printouts."""
    fn = sigmoid if act_name == "σ" else math.tanh
    print(f"\n  [{name}] Shape: (2x4) @ (4x1) + (2x1) -> (2x1)")
    pre_vec = []
    out_vec = []
    for row_idx in range(H):
        terms = [round(W[row_idx][col_idx] * z[col_idx], 6) for col_idx in range(H + E_DIM)]
        dot_val = r4(sum(terms))
        pre_val = r4(dot_val + b[row_idx])
        act_val = r4(fn(pre_val))
        pre_vec.append(pre_val)
        out_vec.append(act_val)
        terms_str = " + ".join(f"({W[row_idx][c]})({z[c]})" for c in range(H + E_DIM))
        print(
            f"    Row {row_idx+1}: {terms_str} + ({b[row_idx]})"
            f" = {dot_val:.4f} + {b[row_idx]:.4f} = {pre_val:.4f}"
            f" --[{act_name}]--> {act_val:.4f}"
        )
    return out_vec


def run_handwritten_trace():
    print("=" * 84)
    print(" HANDWRITTEN ASSIGNMENT TRACE: LSTM FOR NEXT-WORD GENERATION")
    print(f" Settings: V={V}, h={H}, e={E_DIM}, c={C_LEN} | Context: [0, 1, 2] ('I love deep')")
    print("=" * 84)

    context_ids = [0, 1, 2]

    print("\n--- STEP 1: TOKEN EMBEDDING LOOKUP (Shape: 2x1 per token) ---")
    for idx, tok_id in enumerate(context_ids, 1):
        print(f"  x_{idx} (Token ID {tok_id}: '{VOCAB[tok_id]}') = {E_MATRIX[tok_id]}  [Shape: (2x1)]")

    h_prev = [0.0, 0.0]
    C_prev = [0.0, 0.0]
    h_exact = [0.0, 0.0]
    C_exact = [0.0, 0.0]

    for t, tok_id in enumerate(context_ids, 1):
        x_t = E_MATRIX[tok_id]
        z_t = h_prev + x_t  # Shape (4x1)

        print("\n" + "=" * 84)
        print(f" STEP 2.{t}: LSTM TIME STEP t = {t} | Token ID = {tok_id} ('{VOCAB[tok_id]}')")
        print("=" * 84)
        print(f"  Concatenated [h_{t-1}, x_{t}] (Shape 4x1) = {z_t}")

        f_t = compute_gate_explicit(f"Forget Gate f_{t}", W_f, z_t, b_f, "σ")
        i_t = compute_gate_explicit(f"Input Gate i_{t}", W_i, z_t, b_i, "σ")
        o_t = compute_gate_explicit(f"Output Gate o_{t}", W_o, z_t, b_o, "σ")
        c_tilde = compute_gate_explicit(f"Candidate C_tilde_{t}", W_c, z_t, b_c, "tanh")

        f_C = [r4(f_t[k] * C_prev[k]) for k in range(H)]
        i_Ct = [r4(i_t[k] * c_tilde[k]) for k in range(H)]
        C_t = [r4(f_C[k] + i_Ct[k]) for k in range(H)]
        tanh_C = [r4(math.tanh(C_t[k])) for k in range(H)]
        h_t = [r4(o_t[k] * tanh_C[k]) for k in range(H)]

        print(f"\n  [Cell State C_{t}] Shape: (2x1)")
        print(f"    f_{t} ⊙ C_{t-1}     = {f_C}")
        print(f"    i_{t} ⊙ C_tilde_{t} = {i_Ct}")
        print(f"    C_{t}               = {C_t}")

        print(f"\n  [Hidden State h_{t}] Shape: (2x1)")
        print(f"    tanh(C_{t})         = {tanh_C}")
        print(f"    h_{t} = o_{t} ⊙ tanh(C_{t}) = {h_t}")

        h_prev, C_prev = h_t, C_t

        # Track unrounded 64-bit IEEE float in parallel for precision audit
        z_ex = h_exact + x_t
        f_ex = [sigmoid(sum(W_f[r][c] * z_ex[c] for c in range(4)) + b_f[r]) for r in range(2)]
        i_ex = [sigmoid(sum(W_i[r][c] * z_ex[c] for c in range(4)) + b_i[r]) for r in range(2)]
        o_ex = [sigmoid(sum(W_o[r][c] * z_ex[c] for c in range(4)) + b_o[r]) for r in range(2)]
        ct_ex = [math.tanh(sum(W_c[r][c] * z_ex[c] for c in range(4)) + b_c[r]) for r in range(2)]
        C_exact = [f_ex[k] * C_exact[k] + i_ex[k] * ct_ex[k] for k in range(2)]
        h_exact = [o_ex[k] * math.tanh(C_exact[k]) for k in range(2)]

    # Step 3: Dense Logits Layer
    print("\n" + "=" * 84)
    print(" STEP 3: LOGITS LAYER y = W_out @ h_3 + b_out  [Shape: (5x2) @ (2x1) + (5x1) = (5x1)]")
    print("=" * 84)
    y = []
    for k in range(V):
        t0 = round(W_out[k][0] * h_prev[0], 6)
        t1 = round(W_out[k][1] * h_prev[1], 6)
        yk = r4(t0 + t1 + b_out[k])
        y.append(yk)
        print(
            f"  y_{k} ('{VOCAB[k]:<8s}'): ({W_out[k][0]})({h_prev[0]:.4f}) + "
            f"({W_out[k][1]})({h_prev[1]:.4f}) + ({b_out[k]}) = {yk:.4f}"
        )

    # Step 4: Softmax & Token Prediction
    print("\n" + "=" * 84)
    print(" STEP 4: SOFTMAX PROBABILITIES & FINAL NEXT-WORD PREDICTION  [Shape: (5x1)]")
    print("=" * 84)
    exp_y = [r4(math.exp(v)) for v in y]
    sum_exp = r4(sum(exp_y))
    probs = [r4(v / sum_exp) for v in exp_y]

    # Exact float probabilities for audit
    y_ex = [sum(W_out[k][c] * h_exact[c] for c in range(2)) + b_out[k] for k in range(V)]
    exp_ex = [math.exp(v) for v in y_ex]
    probs_ex = [r4(v / sum(exp_ex)) for v in exp_ex]

    print(f"  Partition Sum Σ exp(y_j) = {sum_exp:.4f}\n")
    for k in range(V):
        marker = "  <-- HIGHEST PROBABILITY (PREDICTED NEXT TOKEN)" if k == probs.index(max(probs)) else ""
        print(
            f"  Token ID {k} ('{VOCAB[k]:<8s}'): exp({y[k]:7.4f}) = {exp_y[k]:.4f} "
            f"--> P_{k} = {probs[k]:.4f} ({probs[k]*100:5.2f}%) [64-bit exact: {probs_ex[k]:.4f}]{marker}"
        )

    best_id = probs.index(max(probs))
    print("\n" + "-" * 84)
    print(f"  Sum of Probabilities : {sum(probs):.4f}")
    print(f"  Predicted Token ID   : {best_id} ('{VOCAB[best_id]}')")
    print(f"  Completed Sequence   : 'I love deep {VOCAB[best_id]}'")
    print("=" * 84)


if __name__ == "__main__":
    run_handwritten_trace()
