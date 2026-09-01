"""
Recurrent Neural Network (RNN) from Scratch in Pure Python (Zero external dependencies).
Demonstrates:
1. Exact mathematical forward pass matching standard equations:
   h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
   y_t = softmax(W_hy * h_t + b_y)
2. Manual unrolling over sequential tokens.
3. Information bottleneck & gradient decay demonstration.
"""

import math
import random


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
    """Applies tanh non-linearity element-wise."""
    return [[math.tanh(M[i][j]) for j in range(len(M[0]))] for i in range(len(M))]


def softmax(logits: list[list[float]]) -> list[list[float]]:
    """Numerically stable softmax for column vector (V x 1)."""
    max_val = max(row[0] for row in logits)
    exp_vals = [math.exp(row[0] - max_val) for row in logits]
    sum_exp = sum(exp_vals)
    return [[val / sum_exp] for val in exp_vals]


class PurePythonRNN:
    """
    Vanilla RNN Cell and Sequence Unroller implemented without external libraries.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, seed: int = 42):
        random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Xavier initialization scaling
        scale_xh = 1.0 / math.sqrt(input_dim)
        scale_hh = 1.0 / math.sqrt(hidden_dim)
        scale_hy = 1.0 / math.sqrt(hidden_dim)

        # W_xh: (hidden_dim x input_dim)
        self.W_xh = [[random.uniform(-scale_xh, scale_xh) for _ in range(input_dim)] for _ in range(hidden_dim)]
        
        # W_hh: (hidden_dim x hidden_dim)
        self.W_hh = [[random.uniform(-scale_hh, scale_hh) for _ in range(hidden_dim)] for _ in range(hidden_dim)]
        
        # b_h: (hidden_dim x 1)
        self.b_h = [[0.0] for _ in range(hidden_dim)]

        # W_hy: (output_dim x hidden_dim)
        self.W_hy = [[random.uniform(-scale_hy, scale_hy) for _ in range(hidden_dim)] for _ in range(output_dim)]
        
        # b_y: (output_dim x 1)
        self.b_y = [[0.0] for _ in range(output_dim)]

    def step_forward(self, x_t: list[list[float]], h_prev: list[list[float]]):
        """
        Calculates one time-step:
          1. input_contribution = W_xh * x_t
          2. memory_contribution = W_hh * h_{t-1}
          3. h_t = tanh(input_contrib + memory_contrib + b_h)
          4. logits = W_hy * h_t + b_y
          5. y_t = softmax(logits)
        """
        input_contrib = matmul(self.W_xh, x_t)
        memory_contrib = matmul(self.W_hh, h_prev)
        linear_h = mat_add(mat_add(input_contrib, memory_contrib), self.b_h)
        
        h_t = apply_tanh(linear_h)

        logits = mat_add(matmul(self.W_hy, h_t), self.b_y)
        y_prob = softmax(logits)

        return h_t, y_prob

    def forward_sequence(self, sequence_inputs: list[list[list[float]]]):
        """
        Unrolls RNN over sequence of T inputs.
        """
        # h_0 initialized to all zeros
        h_0 = [[0.0] for _ in range(self.hidden_dim)]
        
        hidden_states = [h_0]
        outputs = []

        h_curr = h_0
        for x_t in sequence_inputs:
            h_curr, y_prob = self.step_forward(x_t, h_curr)
            hidden_states.append(h_curr)
            outputs.append(y_prob)

        return hidden_states, outputs


# =====================================================================
# DEMO WALKTHROUGH
# =====================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" RECURRENT NEURAL NETWORK (RNN) - STEP-BY-STEP FORWARD PASS")
    print("=" * 70)

    # 1. Setup Toy Vocabulary and Embeddings
    vocab = ["The", "chef", "cooked", "the", "delicious", "meal"]
    vocab_size = len(vocab)
    embedding_dim = 4
    hidden_dim = 5  # Size of internal memory state

    random.seed(100)
    # Generate fixed random embeddings (dim x 1)
    embeddings = {
        word: [[random.gauss(0, 1)] for _ in range(embedding_dim)]
        for word in vocab
    }

    # 2. Instantiate RNN Cell
    rnn = PurePythonRNN(input_dim=embedding_dim, hidden_dim=hidden_dim, output_dim=vocab_size)

    # 3. Input Sentence
    sentence = ["The", "chef", "cooked", "the", "delicious"]
    sequence_inputs = [embeddings[w] for w in sentence]

    # 4. Process Sequence
    hidden_states, predictions = rnn.forward_sequence(sequence_inputs)

    print(f"\nSentence: {' -> '.join(sentence)}")
    print(f"Memory dimension (Hidden State Size): {hidden_dim} neurons\n")

    for t, word in enumerate(sentence):
        h_t = hidden_states[t + 1]
        h_vals = [round(row[0], 3) for row in h_t]
        
        # Calculate vector magnitude (Euclidean norm)
        norm = math.sqrt(sum(row[0] ** 2 for row in h_t))
        
        print(f"Step t={t+1} | Token: '{word:9s}' | Hidden State h_{t+1} vector: {h_vals} (norm: {norm:.3f})")

    print("\n" + "=" * 70)
    print(" THE PRESSURE ON THE LAST HIDDEN LAYER (h_T / Information Bottleneck)")
    print("=" * 70)
    last_hidden = [round(row[0], 3) for row in hidden_states[-1]]
    print(f"Final summary vector h_5: {last_hidden}")
    print("\nKey Takeaways on RNN Architecture & Failure Modes:")
    print("1. COMPRESSION PRESSURE: The meaning of all 5 tokens was squeezed into")
    print(f"   these {hidden_dim} numbers. In a 100-word paragraph, the final h_T is forced")
    print("   to memorize everything, leading to severe information loss.")
    print("2. CATASTROPHIC FORGETTING: Early words ('The', 'chef') get repeatedly")
    print("   multiplied by W_hh and squashed by tanh at every single step.")
    print("3. SEQUENTIAL BOTTLENECK: Notice step t=4 CANNOT run until t=3 finishes.")
    print("   This prevented RNNs from scaling on modern parallel GPUs (unlike Transformers).")
    print("=" * 70)
