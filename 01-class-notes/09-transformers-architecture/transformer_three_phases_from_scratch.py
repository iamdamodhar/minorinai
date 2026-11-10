"""
Transformers Architecture: Complete 3-Phase Microscopic Trace from Scratch (Pure Python)
========================================================================================
Zero-dependency standard-library implementation tracing every mathematical operation across:

  - Part 0  : Word2Vec Vector Space Geometry (`Live_Vector_embedding.ipynb` math:
              King - Man + Woman = Queen, Cosine Similarity vs. Euclidean L2 Distance)
  - Phase 1 : INPUT PIPELINE
              Tokenized Text -> Token Embedding (E_tok) + Positional Embedding (E_pos)
  - Phase 2 : PROCESSOR (Pre-LN Transformer Block)
              LayerNorm 1 -> Multi-Head Causal Self-Attention -> Dropout -> Residual (+)
              -> LayerNorm 2 -> Feed-Forward Network (GELU, 4 -> 8 -> 4) -> Dropout -> Residual (+)
  - Phase 3 : OUTPUT PIPELINE
              Final LayerNorm -> Output Linear Layer (Logits) -> Temperature Softmax
"""

import math


# ============================================================================
# VECTOR & MATRIX PRIMITIVES (ZERO EXTERNAL DEPENDENCIES)
# ============================================================================

def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def l2_norm(v: list[float]) -> float:
    return math.sqrt(dot(v, v))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    denom = l2_norm(a) * l2_norm(b)
    return dot(a, b) / denom if denom > 1e-12 else 0.0


def euclidean_distance(a: list[float], b: list[float]) -> float:
    return l2_norm([x - y for x, y in zip(a, b)])


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Multiply A (m x k) by B (k x n) -> (m x n)."""
    m, k, n = len(A), len(A[0]), len(B[0])
    return [[sum(A[i][p] * B[p][j] for p in range(k)) for j in range(n)] for i in range(m)]


def transpose(M: list[list[float]]) -> list[list[float]]:
    return [[M[r][c] for r in range(len(M))] for c in range(len(M[0]))]


def softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    scaled = [s / temperature for s in scores]
    finite = [s for s in scaled if s > -1e8]
    m = max(finite) if finite else 0.0
    exps = [0.0 if s <= -1e8 else math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def gelu(x: float) -> float:
    """Gaussian Error Linear Unit: 0.5 * x * (1 + erf(x / sqrt(2)))."""
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def layer_norm_row(x: list[float], gamma: list[float], beta: list[float], eps: float = 1e-5) -> tuple[list[float], float, float]:
    """Normalizes a single token vector x in R^{d_model} across its feature dimensions."""
    d = len(x)
    mean = sum(x) / d
    var = sum((v - mean) ** 2 for v in x) / d
    std = math.sqrt(var + eps)
    normed = [gamma[i] * ((x[i] - mean) / std) + beta[i] for i in range(d)]
    return normed, mean, var


def layer_norm(X: list[list[float]], gamma: list[float], beta: list[float], eps: float = 1e-5) -> list[list[float]]:
    return [layer_norm_row(row, gamma, beta, eps)[0] for row in X]


def dropout_matrix(X: list[list[float]], drop_prob: float, mask: list[list[int]] | None = None, training: bool = True) -> list[list[float]]:
    """Inverted Dropout: scales kept activations by 1 / (1 - p) during training; identity at inference."""
    if not training or drop_prob <= 0.0:
        return [row[:] for row in X]
    scale = 1.0 / (1.0 - drop_prob)
    return [
        [X[r][c] * scale if (mask is None or mask[r][c] == 1) else 0.0 for c in range(len(X[0]))]
        for r in range(len(X))
    ]


def fmt_mat(M: list[list[float]], indent: str = "    ") -> str:
    return "\n".join(f"{indent}[" + ", ".join(f"{v:+7.4f}" for v in row) + "]" for row in M)


# ============================================================================
# PART 0: VECTOR EMBEDDING GEOMETRY (MIRRORING `Live_Vector_embedding.ipynb`)
# ============================================================================

def run_part0_word2vec_geometry_demo() -> None:
    print("=" * 92)
    print(" PART 0: WORD EMBEDDING GEOMETRY (`Live_Vector_embedding.ipynb` FIRST-PRINCIPLES MATH)")
    print("         4D Semantic Space: [Royalty, Femininity, Maturity/Age, Tech-vs-Organic]")
    print("=" * 92)

    vectors = {
        "king":          [ 0.95, -0.80,  0.70,  0.00],
        "man":           [ 0.05, -0.85,  0.65,  0.00],
        "woman":         [ 0.05,  0.85,  0.65,  0.00],
        "queen":         [ 0.95,  0.88,  0.68,  0.00],
        "semiconductor": [ 0.00,  0.00,  0.10,  0.98],
        "earthworm":     [ 0.00,  0.00, -0.20, -0.96],
    }

    # 1. King - Man + Woman analogy
    analogy_vec = [
        vectors["king"][d] - vectors["man"][d] + vectors["woman"][d]
        for d in range(4)
    ]
    sim_to_queen = cosine_similarity(analogy_vec, vectors["queen"])
    print(f"  v('king') - v('man') + v('woman') = {[round(v, 3) for v in analogy_vec]}")
    print(f"  v('queen')                        = {vectors['queen']}")
    print(f"  --> Cosine Similarity to 'queen'  = {sim_to_queen:.4f} (Notebook Cell 8: 0.7118)")

    # 2. Cosine Similarity vs. Euclidean Distance (Cells 10 & 14)
    pairs = [("man", "woman"), ("king", "queen"), ("semiconductor", "earthworm")]
    print("-" * 92)
    for w1, w2 in pairs:
        cs = cosine_similarity(vectors[w1], vectors[w2])
        ed = euclidean_distance(vectors[w1], vectors[w2])
        print(f"  Pair ({w1:<13s}, {w2:<11s}) | Cosine Sim = {cs:+7.4f} | Euclidean L2 Dist ||v1-v2|| = {ed:.4f}")


# ============================================================================
# PHASE 1, PHASE 2, PHASE 3: COMPLETE TRANSFORMER FORWARD PASS
# ============================================================================

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> list[list[float]]:
    """
    Vaswani et al. (2017) Sinusoidal Positional Encoding:
      PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    """
    pe = []
    for pos in range(seq_len):
        row = []
        for i in range(0, d_model, 2):
            div_term = math.pow(10000.0, i / d_model)
            row.append(math.sin(pos / div_term))
            if i + 1 < d_model:
                row.append(math.cos(pos / div_term))
        pe.append(row)
    return pe


def run_three_phase_transformer_trace() -> None:
    vocab = ["<pad>", "myntra", "linen", "kurta", "is", "breathable"]
    word2id = {w: idx for idx, w in enumerate(vocab)}
    d_model = 4
    num_heads = 2
    d_k = d_model // num_heads  # 2 dimensions per head

    # ========================================================================
    # PHASE 1: INPUT PIPELINE (Tokenized Text -> Token Emb + Positional Emb)
    # ========================================================================
    print("\n" + "=" * 92)
    print(" PHASE 1: INPUT PIPELINE (Tokenized Text -> Token Embedding + Positional Embedding)")
    print("=" * 92)

    tokens = ["myntra", "linen", "kurta"]
    token_ids = [word2id[t] for t in tokens]
    T = len(token_ids)
    print(f"  Step 1.1 Tokenized Text : {tokens}  -->  Token IDs = {token_ids} (Shape: [T={T}])")

    # Embedding Lookup Table E in R^{V=6 x d_model=4}
    # Axes: [Brand/Entity, Fabric-Summer, Garment-Formality, Linguistic-Syntax]
    E_table = [
        [ 0.00,  0.00,  0.00,  0.00],  # 0: <pad>
        [ 0.90,  0.20,  0.40, -0.10],  # 1: myntra
        [ 0.10,  0.95,  0.30,  0.20],  # 2: linen
        [ 0.20,  0.60,  0.95,  0.10],  # 3: kurta
        [-0.20,  0.00,  0.10,  0.90],  # 4: is
        [ 0.15,  0.92,  0.75,  0.40],  # 5: breathable
    ]

    E_tok = [E_table[idx][:] for idx in token_ids]
    E_pos = sinusoidal_positional_encoding(seq_len=T, d_model=d_model)
    X_0 = [[E_tok[t][d] + E_pos[t][d] for d in range(d_model)] for t in range(T)]

    print(f"\n  Step 1.2 Token Embeddings E_tok (Shape: [{T}, {d_model}]):\n{fmt_mat(E_tok)}")
    print(f"\n  Step 1.3 Sinusoidal Positional Embeddings E_pos (Shape: [{T}, {d_model}]):\n{fmt_mat(E_pos)}")
    print(f"\n  Step 1.4 Combined Input X^(0) = E_tok + E_pos (Shape: [{T}, {d_model}]):\n{fmt_mat(X_0)}")

    # ========================================================================
    # PHASE 2: THE PROCESSOR (Pre-LN Transformer Block)
    # ========================================================================
    print("\n" + "=" * 92)
    print(" PHASE 2: PROCESSOR BLOCK (LN1 -> Multi-Head Attention -> Dropout -> Res1 -> LN2 -> FFN -> Res2)")
    print("=" * 92)

    gamma = [1.0] * d_model
    beta = [0.0] * d_model

    # Step 2.1: LayerNorm 1
    Z_1 = layer_norm(X_0, gamma, beta)
    print(f"  Step 2.1 LayerNorm 1 Output Z^(1) = LN_1(X^(0)) (Mean=0, Var=1 per token):\n{fmt_mat(Z_1)}")

    # Step 2.2: Multi-Head Causal Self-Attention (2 Heads, d_k = 2 each)
    # Head 1 projections (focusing on Brand/Fabric compatibility)
    W_Q1 = [[0.8, 0.1], [0.2, 0.9], [0.1, 0.2], [0.0, 0.1]]
    W_K1 = [[0.7, 0.2], [0.3, 0.8], [0.2, 0.1], [0.1, 0.0]]
    W_V1 = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.0, 0.0]]

    # Head 2 projections (focusing on Garment/Syntax structure)
    W_Q2 = [[0.1, 0.0], [0.2, 0.1], [0.9, 0.2], [0.1, 0.8]]
    W_K2 = [[0.0, 0.1], [0.1, 0.3], [0.8, 0.2], [0.2, 0.9]]
    W_V2 = [[0.0, 0.0], [0.4, 0.2], [1.0, 0.0], [0.0, 1.0]]

    W_O = [
        [0.6, 0.2, 0.2, 0.0],
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.1, 0.7, 0.0],
        [0.0, 0.1, 0.2, 0.8],
    ]

    head_outputs = []
    for h_idx, (WQ, WK, WV) in enumerate([(W_Q1, W_K1, W_V1), (W_Q2, W_K2, W_V2)], 1):
        Q = matmul(Z_1, WQ)  # [3, 2]
        K = matmul(Z_1, WK)  # [3, 2]
        V = matmul(Z_1, WV)  # [3, 2]
        raw_scores = matmul(Q, transpose(K))  # [3, 3]
        # Scale by 1 / sqrt(d_k) and apply Causal Mask (j > i -> -1e9)
        attn_probs = []
        for i in range(T):
            masked_row = [
                (raw_scores[i][j] / math.sqrt(d_k)) if j <= i else -1e9
                for j in range(T)
            ]
            attn_probs.append(softmax(masked_row))
        head_out = matmul(attn_probs, V)  # [3, 2]
        head_outputs.append(head_out)
        print(f"\n  Step 2.2.{h_idx} Head {h_idx} Causal Attention Weights A_{h_idx} (Shape: [{T}, {T}]):\n{fmt_mat(attn_probs)}")

    # Concatenate Head 1 [3, 2] and Head 2 [3, 2] -> [3, 4] and project via W_O
    concat_heads = [head_outputs[0][t] + head_outputs[1][t] for t in range(T)]
    MHA_out = matmul(concat_heads, W_O)
    print(f"\n  Step 2.2.3 Multi-Head Attention Projected Output MHA(Z^(1)) (Shape: [{T}, {d_model}]):\n{fmt_mat(MHA_out)}")

    # Step 2.3: Dropout 1 (inference/deterministic pass) + Residual Add 1
    MHA_drop = dropout_matrix(MHA_out, drop_prob=0.1, training=False)
    X_1 = [[X_0[t][d] + MHA_drop[t][d] for d in range(d_model)] for t in range(T)]
    print(f"\n  Step 2.3 Residual Stream 1: X^(1) = X^(0) + Dropout(MHA(LN_1(X^(0)))):\n{fmt_mat(X_1)}")

    # Step 2.4: LayerNorm 2
    Z_2 = layer_norm(X_1, gamma, beta)
    print(f"\n  Step 2.4 LayerNorm 2 Output Z^(2) = LN_2(X^(1)):\n{fmt_mat(Z_2)}")

    # Step 2.5: Position-Wise Feed-Forward Network (d_model=4 -> d_ff=8 -> d_model=4 with GELU)
    W_ff1 = [
        [ 0.5, -0.3,  0.4,  0.2, -0.4,  0.6,  0.1, -0.2],
        [ 0.2,  0.7, -0.2,  0.5,  0.3, -0.1,  0.4,  0.3],
        [-0.3,  0.2,  0.8, -0.1,  0.5,  0.2, -0.3,  0.6],
        [ 0.1, -0.4,  0.2,  0.6, -0.2,  0.3,  0.5, -0.1],
    ]  # [4, 8]
    b_ff1 = [0.05] * 8
    W_ff2 = transpose(W_ff1)  # [8, 4]
    b_ff2 = [0.0] * d_model

    FFN_hidden = [
        [gelu(val + b_ff1[c]) for c, val in enumerate(row)]
        for row in matmul(Z_2, W_ff1)
    ]  # [3, 8]
    FFN_out = [
        [val + b_ff2[c] for c, val in enumerate(row)]
        for row in matmul(FFN_hidden, W_ff2)
    ]  # [3, 4]
    print(f"\n  Step 2.5 Feed-Forward Network Output FFN(Z^(2)) (4 -> 8 GELU -> 4):\n{fmt_mat(FFN_out)}")

    # Step 2.6: Dropout 2 + Residual Add 2
    FFN_drop = dropout_matrix(FFN_out, drop_prob=0.1, training=False)
    X_2 = [[X_1[t][d] + FFN_drop[t][d] for d in range(d_model)] for t in range(T)]
    print(f"\n  Step 2.6 Residual Stream 2: X^(2) = X^(1) + Dropout(FFN(LN_2(X^(1)))):\n{fmt_mat(X_2)}")

    # ========================================================================
    # PHASE 3: OUTPUT PIPELINE (Final LayerNorm -> Output Linear Head -> Softmax)
    # ========================================================================
    print("\n" + "=" * 92)
    print(" PHASE 3: OUTPUT PIPELINE (Final LayerNorm -> Output Linear Projection -> Next-Token Softmax)")
    print("=" * 92)

    # Step 3.1: Final LayerNorm
    H_final = layer_norm(X_2, gamma, beta)
    print(f"  Step 3.1 Final LayerNorm Output H_final = LN_final(X^(2)) (Shape: [{T}, {d_model}]):\n{fmt_mat(H_final)}")

    # Step 3.2: Output Layer (Weight-Tied / Linear Projection H_final @ E_table^T + bias -> [T, V=6])
    b_out = [-1.0, -0.2, -0.1, -0.1, +0.9, +1.1]
    raw_logits = matmul(H_final, transpose(E_table))  # [3, 6]
    logits = [[raw_logits[t][v] + b_out[v] for v in range(len(vocab))] for t in range(T)]

    # Next-token probability distribution after final token t=2 ("kurta")
    last_logits = logits[-1]
    probs = softmax(last_logits, temperature=1.0)
    print(f"\n  Step 3.2 Output Layer Vocabulary Probabilities after 'myntra linen kurta' (t=2):")
    print("-" * 92)
    for v_idx, word in enumerate(vocab):
        bar = "█" * int(round(probs[v_idx] * 30))
        print(f"    Token {v_idx} ({word:<10s}) | Logit = {last_logits[v_idx]:+7.4f} | Prob = {probs[v_idx]*100:5.2f}% {bar}")
    best_id = max(range(len(vocab)), key=lambda i: probs[i])
    print(f"  --> Predicted Next Token: '{vocab[best_id]}' ({probs[best_id]*100:.2f}%)")
    print("=" * 92)


if __name__ == "__main__":
    run_part0_word2vec_geometry_demo()
    run_three_phase_transformer_trace()
