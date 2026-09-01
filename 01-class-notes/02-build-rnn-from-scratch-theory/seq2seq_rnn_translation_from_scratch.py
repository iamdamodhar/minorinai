"""
Complete Seq2Seq (Encoder-Decoder) RNN from Scratch in Pure Python.
Demonstrates:
1. English-to-Hindi translation: "You are good" -> "आप अच्छे हैं"
2. Random parameter initialization for Encoder & Decoder matrices.
3. Explicit step-by-step matrix computations:
   - Encoder: h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
   - Context Vector hand-off: s_0 = h_enc_T
   - Decoder: s_t = tanh(W_ys @ y_{t-1} + W_ss @ s_{t-1} + b_s)
   - Logits & Softmax: z_t = W_vocab @ s_t + b_vocab -> P_t = Softmax(z_t)
4. Numerical Stability (Max-Subtraction trick) in Softmax.
5. Modern GPT Temperature Scaling demonstration.
"""

import math
import random


# =====================================================================
# MATRIX MATH HELPERS (Pure Python Standard Library)
# =====================================================================
def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Matrix multiplication: A (m x k) @ B (k x n) -> (m x n)"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, f"Shape mismatch: {cols_A} != {rows_B}"

    result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
    for i in range(rows_A):
        for j in range(cols_B):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(cols_A))
    return result


def mat_add(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """Element-wise matrix addition: A + B"""
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]


def apply_tanh(M: list[list[float]]) -> list[list[float]]:
    """Element-wise hyperbolic tangent activation."""
    return [[math.tanh(M[i][j]) for j in range(len(M[0]))] for i in range(len(M))]


def stable_softmax(logits: list[list[float]], temperature: float = 1.0) -> list[list[float]]:
    """
    Numerically stable Softmax with temperature scaling:
      P_i = exp((z_i - max(z)) / T) / sum(exp((z_j - max(z)) / T))
    """
    scaled_logits = [row[0] / temperature for row in logits]
    max_val = max(scaled_logits)
    exp_vals = [math.exp(val - max_val) for val in scaled_logits]
    sum_exp = sum(exp_vals)
    return [[val / sum_exp] for val in exp_vals]


# =====================================================================
# ENCODER-DECODER SEQ2SEQ RNN CLASS
# =====================================================================
class Seq2SeqRNN:
    def __init__(self, emb_dim: int, hidden_dim: int, src_vocab_size: int, tgt_vocab_size: int, seed: int = 42):
        random.seed(seed)
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        scale = 1.0 / math.sqrt(hidden_dim)

        # -------------------------------------------------------------
        # 1. ENCODER PARAMETERS
        # -------------------------------------------------------------
        # W_xh_enc: (hidden_dim x emb_dim)
        self.W_xh_enc = [[random.uniform(-scale, scale) for _ in range(emb_dim)] for _ in range(hidden_dim)]
        # W_hh_enc: (hidden_dim x hidden_dim)
        self.W_hh_enc = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        # b_h_enc: (hidden_dim x 1)
        self.b_h_enc = [[0.0] for _ in range(hidden_dim)]

        # -------------------------------------------------------------
        # 2. DECODER PARAMETERS
        # -------------------------------------------------------------
        # W_ys_dec: (hidden_dim x emb_dim)
        self.W_ys_dec = [[random.uniform(-scale, scale) for _ in range(emb_dim)] for _ in range(hidden_dim)]
        # W_ss_dec: (hidden_dim x hidden_dim)
        self.W_ss_dec = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        # b_s_dec: (hidden_dim x 1)
        self.b_s_dec = [[0.0] for _ in range(hidden_dim)]

        # Output Projection Layer (to Target Vocabulary Logits)
        # W_vocab: (tgt_vocab_size x hidden_dim)
        self.W_vocab = [[random.uniform(-scale, scale) for _ in range(hidden_dim)] for _ in range(tgt_vocab_size)]
        # b_vocab: (tgt_vocab_size x 1)
        self.b_vocab = [[0.0] for _ in range(tgt_vocab_size)]

    def encode(self, source_embeddings: list[list[list[float]]]):
        """
        Runs the Encoder RNN across the source English sequence.
        Returns:
          hidden_states: list of all h_t states
          context_vector: final hidden state h_T
        """
        # Initialize h_0 to zeros (hidden_dim x 1)
        h_curr = [[0.0] for _ in range(self.hidden_dim)]
        all_hidden = [h_curr]

        for x_t in source_embeddings:
            # h_t = tanh(W_xh @ x_t + W_hh @ h_{t-1} + b_h)
            linear_input = matmul(self.W_xh_enc, x_t)
            linear_memory = matmul(self.W_hh_enc, h_curr)
            linear_sum = mat_add(mat_add(linear_input, linear_memory), self.b_h_enc)
            h_curr = apply_tanh(linear_sum)
            all_hidden.append(h_curr)

        context_vector = h_curr
        return all_hidden, context_vector

    def decode_step(self, y_prev_emb: list[list[float]], s_prev: list[list[float]], temperature: float = 1.0):
        """
        Runs a single step of the Decoder RNN.
        Computes:
          1. s_t = tanh(W_ys @ y_{t-1} + W_ss @ s_{t-1} + b_s)
          2. logits = W_vocab @ s_t + b_vocab
          3. probs = Softmax(logits / T)
        """
        linear_input = matmul(self.W_ys_dec, y_prev_emb)
        linear_memory = matmul(self.W_ss_dec, s_prev)
        linear_sum = mat_add(mat_add(linear_input, linear_memory), self.b_s_dec)
        s_t = apply_tanh(linear_sum)

        # Compute raw unnormalized logits
        logits = mat_add(matmul(self.W_vocab, s_t), self.b_vocab)
        
        # Softmax probability distribution
        probs = stable_softmax(logits, temperature=temperature)

        return s_t, logits, probs


# =====================================================================
# DEMO WALKTHROUGH: "You are good" -> "आप अच्छे हैं"
# =====================================================================
if __name__ == "__main__":
    print("=" * 75)
    print(" BUILD RNN FROM SCRATCH THEORY: SEQ2SEQ ENGLISH-TO-HINDI TRANSLATION")
    print("=" * 75)

    # 1. Vocabularies
    src_vocab = {"<PAD>": 0, "You": 1, "are": 2, "good": 3, "<EOS>": 4}
    tgt_vocab = {"<PAD>": 0, "<SOS>": 1, "आप": 2, "अच्छे": 3, "हैं": 4, "<EOS>": 5}
    idx2tgt = {v: k for k, v in tgt_vocab.items()}

    emb_dim = 4      # Embedding dimension (d_in)
    hidden_dim = 3   # Hidden state dimension (d_h)

    # Deterministic dummy embeddings (emb_dim x 1)
    random.seed(123)
    src_embeddings = {w: [[random.gauss(0, 1)] for _ in range(emb_dim)] for w in src_vocab}
    tgt_embeddings = {w: [[random.gauss(0, 1)] for _ in range(emb_dim)] for w in tgt_vocab}

    # Instantiate Seq2Seq model
    model = Seq2SeqRNN(
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        seed=42
    )

    # =================================================================
    # STEP 1: ENCODER FORWARD PASS
    # =================================================================
    input_sentence = ["You", "are", "good"]
    print(f"\n[ENCODER] Source English Input: {' '.join(input_sentence)}")
    print(f"Matrix Shapes: W_xh=({hidden_dim}x{emb_dim}), W_hh=({hidden_dim}x{hidden_dim}), b_h=({hidden_dim}x1)\n")

    encoder_inputs = [src_embeddings[w] for w in input_sentence]
    enc_hidden_states, context_vector = model.encode(encoder_inputs)

    for t, word in enumerate(input_sentence):
        h_vals = [round(r[0], 4) for r in enc_hidden_states[t + 1]]
        print(f"  Encoder Step t={t+1} | Token: '{word:5s}' ──► Hidden State h_{t+1}: {h_vals}")

    c_vals = [round(r[0], 4) for r in context_vector]
    print(f"\n[CONTEXT VECTOR (c)] Hand-off: h_enc_3 = {c_vals}")
    print("  -> All meaning of 'You are good' is compressed into these 3 numbers!")

    # =================================================================
    # STEP 2: DECODER FORWARD PASS (Autoregressive Generation)
    # =================================================================
    print("\n" + "=" * 75)
    print("[DECODER] Autoregressive Generation of Hindi Target:")
    print("=" * 75)

    # Initialize decoder state s_0 with the context vector c
    s_curr = context_vector
    current_token = "<SOS>"
    max_decode_steps = 4

    for t in range(1, max_decode_steps + 1):
        y_emb = tgt_embeddings[current_token]
        s_curr, logits, probs = model.decode_step(y_emb, s_curr, temperature=1.0)

        # Get highest probability token (Argmax)
        flat_logits = [round(r[0], 3) for r in logits]
        flat_probs = [round(r[0], 3) for r in probs]
        best_idx = max(range(len(flat_probs)), key=lambda i: flat_probs[i])
        predicted_word = idx2tgt[best_idx]

        print(f"\nDecode Step t={t}:")
        print(f"  Input Token y_{t-1}      : '{current_token}'")
        print(f"  Decoder State s_{t}      : {[round(r[0], 4) for r in s_curr]}")
        print(f"  Raw Logits (z_{t})       : {flat_logits}")
        print(f"  Softmax Probs (P_{t})    : {flat_probs}")
        print(f"  Predicted Word (Argmax) : '{predicted_word}' (Prob: {flat_probs[best_idx]:.3f})")

        # In a real setup with teacher forcing or trained model, we feed the predicted/target word
        current_token = predicted_word

    # =================================================================
    # STEP 3: SOFTMAX FUNCTION & TEMPERATURE DEEP DIVE (MODERN LLM LINK)
    # =================================================================
    print("\n" + "=" * 75)
    print(" SOFTMAX IN MODERN GENERATIVE AI (GPT/LLM TEMPERATURE DEMO)")
    print("=" * 75)

    sample_logits = [[3.5], [1.2], [0.1], [5.8], [-2.0], [0.5]]
    vocab_labels = ["<PAD>", "<SOS>", "आप", "अच्छे", "हैं", "<EOS>"]

    print(f"Sample Logits z: {[r[0] for r in sample_logits]}")
    for temp in [0.2, 1.0, 2.5]:
        p = stable_softmax(sample_logits, temperature=temp)
        flat_p = [round(r[0], 4) for r in p]
        print(f"  Temperature T={temp:3.1f} ──► Probs: {flat_p}  (Top token '{vocab_labels[3]}': {flat_p[3]*100:.1f}%)")

    print("\nTakeaways on Softmax & Evolution to 2017 Transformers:")
    print("1. SOFTMAX IN ATTENTION: Transformers use Softmax on (Q @ K.T / sqrt(d_k)) to calculate attention weights.")
    print("2. SOFTMAX IN NEXT-TOKEN HEAD: GPT-4/Claude use Softmax on vocabulary logits with Temperature scaling.")
    print("3. WHY RNNs EVOLVED: The 3-number Context Vector bottleneck forced the invention of Bahdanau Attention (2014),")
    print("   which culminated in the 2017 Transformer ('Attention Is All You Need'), eliminating recurrence entirely.")
    print("=" * 75)
