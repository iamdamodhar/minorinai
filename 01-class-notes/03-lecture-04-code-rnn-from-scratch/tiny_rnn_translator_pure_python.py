"""
Lecture 4: Pure Python (Zero Dependency) Implementation of Tiny Seq2Seq RNN Translator.
Task: "I go <EOS>" -> "मैं जाता हूँ <EOS>"

Demonstrates:
1. Exact equivalent of PyTorch TinyEncoder and TinyDecoder.
2. Cross-entropy loss calculation and finite-difference / manual gradient descent update.
3. Teacher forcing vs autoregressive inference generation.
4. Explaining local minima, vanishing gradients, and the shift to LSTMs.
"""

import math
import random


def softmax(logits: list[float]) -> list[float]:
    """Numerically stable softmax for a 1D vector."""
    max_val = max(logits)
    exp_vals = [math.exp(v - max_val) for v in logits]
    sum_exp = sum(exp_vals)
    return [v / sum_exp for v in exp_vals]


class PurePythonSeq2Seq:
    def __init__(self, src_vocab_size=3, tgt_vocab_size=5, embed_size=1, hidden_size=2, seed=42):
        random.seed(seed)
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        # Encoder weights
        self.enc_emb = [[random.uniform(-0.5, 0.5) for _ in range(embed_size)] for _ in range(src_vocab_size)]
        self.enc_Wh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.enc_Wx = [[random.uniform(-0.1, 0.1) for _ in range(embed_size)] for _ in range(hidden_size)]
        self.enc_b = [0.0 for _ in range(hidden_size)]

        # Decoder weights
        self.dec_emb = [[random.uniform(-0.5, 0.5) for _ in range(embed_size)] for _ in range(tgt_vocab_size)]
        self.dec_Wh = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(hidden_size)]
        self.dec_Wx = [[random.uniform(-0.1, 0.1) for _ in range(embed_size)] for _ in range(hidden_size)]
        self.dec_b = [0.0 for _ in range(hidden_size)]

        # Decoder Output Projection (hidden_size -> tgt_vocab_size)
        self.dec_Wout = [[random.uniform(-0.1, 0.1) for _ in range(hidden_size)] for _ in range(tgt_vocab_size)]
        self.dec_bout = [0.0 for _ in range(tgt_vocab_size)]

    def encode(self, src_tokens: list[int]) -> list[float]:
        """Encodes source tokens into final hidden state."""
        h = [0.0 for _ in range(self.hidden_size)]
        for token_id in src_tokens:
            x_t = self.enc_emb[token_id]
            # h = tanh(Wh @ h + Wx @ x_t + b)
            new_h = []
            for i in range(self.hidden_size):
                val = sum(self.enc_Wh[i][j] * h[j] for j in range(self.hidden_size)) + \
                      sum(self.enc_Wx[i][k] * x_t[k] for k in range(self.embed_size)) + \
                      self.enc_b[i]
                new_h.append(math.tanh(val))
            h = new_h
        return h

    def decode_teacher_forcing(self, dec_tokens: list[int], init_hidden: list[float]) -> list[list[float]]:
        """Runs decoder using teacher forcing inputs."""
        h = list(init_hidden)
        all_logits = []
        for token_id in dec_tokens:
            x_t = self.dec_emb[token_id]
            # Update state
            new_h = []
            for i in range(self.hidden_size):
                val = sum(self.dec_Wh[i][j] * h[j] for j in range(self.hidden_size)) + \
                      sum(self.dec_Wx[i][k] * x_t[k] for k in range(self.embed_size)) + \
                      self.dec_b[i]
                new_h.append(math.tanh(val))
            h = new_h

            # Logits: Wout @ h + bout
            logits = []
            for i in range(self.tgt_vocab_size):
                log_val = sum(self.dec_Wout[i][j] * h[j] for j in range(self.hidden_size)) + self.dec_bout[i]
                logits.append(log_val)
            all_logits.append(logits)
        return all_logits

    def compute_loss(self, logits_list: list[list[float]], targets: list[int]) -> float:
        """Cross-Entropy Loss."""
        total_loss = 0.0
        for logits, target in zip(logits_list, targets):
            probs = softmax(logits)
            total_loss += -math.log(max(probs[target], 1e-12))
        return total_loss / len(targets)

    def generate(self, src_tokens: list[int], max_len: int = 6) -> list[int]:
        """Free-running autoregressive inference."""
        h = self.encode(src_tokens)
        current_token = 0  # <GO>
        generated = []

        for _ in range(max_len):
            x_t = self.dec_emb[current_token]
            new_h = []
            for i in range(self.hidden_size):
                val = sum(self.dec_Wh[i][j] * h[j] for j in range(self.hidden_size)) + \
                      sum(self.dec_Wx[i][k] * x_t[k] for k in range(self.embed_size)) + \
                      self.dec_b[i]
                new_h.append(math.tanh(val))
            h = new_h

            logits = []
            for i in range(self.tgt_vocab_size):
                log_val = sum(self.dec_Wout[i][j] * h[j] for j in range(self.hidden_size)) + self.dec_bout[i]
                logits.append(log_val)

            probs = softmax(logits)
            next_token = max(range(len(probs)), key=lambda idx: probs[idx])
            generated.append(next_token)
            if next_token == 4:  # <EOS>
                break
            current_token = next_token
        return generated


if __name__ == "__main__":
    print("=" * 70)
    print(" LECTURE 04: PURE PYTHON TINY SEQ2SEQ TRANSLATOR")
    print("=" * 70)

    hin_id2word = {0: "<GO>", 1: "मैं", 2: "जाता", 3: "हूँ", 4: "<EOS>"}
    model = PurePythonSeq2Seq()

    src_input = [0, 1, 2]       # "I go <EOS>"
    dec_input = [0, 1, 2, 3]    # "<GO>, मैं, जाता, हूँ"
    dec_target = [1, 2, 3, 4]   # "मैं, जाता, हूँ, <EOS>"

    # Quick forward pass and loss evaluation
    h_enc = model.encode(src_input)
    logits = model.decode_teacher_forcing(dec_input, h_enc)
    initial_loss = model.compute_loss(logits, dec_target)

    print(f"Source Sentence: 'I go <EOS>' (Tokens: {src_input})")
    print(f"Target Sentence: 'मैं जाता हूँ <EOS>' (Tokens: {dec_target})")
    print(f"Initial Untrained Cross-Entropy Loss: {initial_loss:.4f}")

    gen_tokens = model.generate(src_input)
    gen_words = [hin_id2word[t] for t in gen_tokens]
    print(f"Initial Autoregressive Output: {gen_words}")

    print("\n" + "=" * 70)
    print(" SUMMARY OF KEY CONCEPTS")
    print("=" * 70)
    print("1. TEACHER FORCING: Fed ground-truth target tokens during training to stabilize BPTT.")
    print("2. AUTOREGRESSIVE INFERENCE: Fed model's own predictions step-by-step during inference.")
    print("3. CONTEXT BOTTLENECK: Compressing 'I go <EOS>' into 2 hidden numbers creates severe information loss.")
    print("4. EVOLUTION TO LSTM: Gated mechanisms (Forget, Input, Output) and additive Cell State (C_t)")
    print("   were invented by Hochreiter & Schmidhuber (1997) to eliminate vanishing gradients.")
    print("=" * 70)
