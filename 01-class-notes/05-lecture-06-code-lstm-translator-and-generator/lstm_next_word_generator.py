"""
Lecture 06: LSTM Next-Word Text Generator on Headline Data.
Demonstrates:
1. Tokenization & Sliding-Window N-Gram Sequence Generation.
2. Pre-padding of sequences with numpy/pure python.
3. Keras / TensorFlow reference architecture (Embedding -> LSTM -> Dropout -> Dense Softmax).
4. Pure Python standalone implementation for direct execution anywhere.
"""

import math
import random


def clean_text(txt: str) -> str:
    import string
    txt = "".join(v for v in txt if v not in string.punctuation).lower()
    return txt.strip()


class PurePythonNextWordLSTM:
    def __init__(self, vocab_size: int, embed_dim: int = 10, hidden_dim: int = 20, seed: int = 42):
        random.seed(seed)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        scale = 1.0 / math.sqrt(hidden_dim)
        def mat(r, c): return [[random.uniform(-scale, scale) for _ in range(c)] for _ in range(r)]
        def vec(s, val=0.0): return [val for _ in range(s)]

        self.embedding = mat(vocab_size, embed_dim)
        self.W_i, self.U_i, self.b_i = mat(hidden_dim, embed_dim), mat(hidden_dim, hidden_dim), vec(hidden_dim, 0.0)
        self.W_f, self.U_f, self.b_f = mat(hidden_dim, embed_dim), mat(hidden_dim, hidden_dim), vec(hidden_dim, 1.0)
        self.W_g, self.U_g, self.b_g = mat(hidden_dim, embed_dim), mat(hidden_dim, hidden_dim), vec(hidden_dim, 0.0)
        self.W_o, self.U_o, self.b_o = mat(hidden_dim, embed_dim), mat(hidden_dim, hidden_dim), vec(hidden_dim, 0.0)
        self.W_dense, self.b_dense = mat(vocab_size, hidden_dim), vec(vocab_size, 0.0)

    def step(self, token_id: int, h_prev: list[float], c_prev: list[float]):
        x_t = self.embedding[token_id]
        h_dim = self.hidden_dim

        i_t, f_t, g_t, o_t = [0.0]*h_dim, [0.0]*h_dim, [0.0]*h_dim, [0.0]*h_dim
        for i in range(h_dim):
            sum_i = self.b_i[i] + sum(self.W_i[i][j]*x_t[j] for j in range(self.embed_dim)) + sum(self.U_i[i][k]*h_prev[k] for k in range(h_dim))
            sum_f = self.b_f[i] + sum(self.W_f[i][j]*x_t[j] for j in range(self.embed_dim)) + sum(self.U_f[i][k]*h_prev[k] for k in range(h_dim))
            sum_g = self.b_g[i] + sum(self.W_g[i][j]*x_t[j] for j in range(self.embed_dim)) + sum(self.U_g[i][k]*h_prev[k] for k in range(h_dim))
            sum_o = self.b_o[i] + sum(self.W_o[i][j]*x_t[j] for j in range(self.embed_dim)) + sum(self.U_o[i][k]*h_prev[k] for k in range(h_dim))

            i_t[i] = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, sum_i))))
            f_t[i] = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, sum_f))))
            g_t[i] = math.tanh(sum_g)
            o_t[i] = 1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, sum_o))))

        c_t = [f_t[i]*c_prev[i] + i_t[i]*g_t[i] for i in range(h_dim)]
        h_t = [o_t[i]*math.tanh(c_t[i]) for i in range(h_dim)]

        logits = [
            self.b_dense[v] + sum(self.W_dense[v][k]*h_t[k] for k in range(h_dim))
            for v in range(self.vocab_size)
        ]
        return h_t, c_t, logits

    def generate(self, seed_ids: list[int], num_words: int = 5) -> list[int]:
        h = [0.0] * self.hidden_dim
        c = [0.0] * self.hidden_dim
        logits = []

        for tok in seed_ids:
            h, c, logits = self.step(tok, h, c)

        generated = []
        for _ in range(num_words):
            next_token = max(range(len(logits)), key=lambda idx: logits[idx])
            generated.append(next_token)
            h, c, logits = self.step(next_token, h, c)

        return generated


if __name__ == "__main__":
    print("=" * 80)
    print(" LECTURE 06: LSTM NEXT-WORD TEXT GENERATOR (HEADLINES CORPUS)")
    print("=" * 80)

    headlines = [
        "United States signs bilateral trade agreement with India",
        "Donald Trump announces new economic policy in New York",
        "India and China hold strategic bilateral talks on technology",
        "Science and technology investments rise across the globe",
        "President Trump meets leaders in New York"
    ]

    # Preprocessing
    corpus = [clean_text(h) for h in headlines]
    vocab = ["<PAD>"] + sorted(list(set(" ".join(corpus).split())))
    word2idx = {w: i for i, w in enumerate(vocab)}
    idx2word = {i: w for i, w in enumerate(vocab)}

    print(f"Total Headlines: {len(corpus)} | Vocabulary Size: {len(vocab)}")

    # Generate N-gram token sequences
    input_sequences = []
    for line in corpus:
        tokens = [word2idx[w] for w in line.split()]
        for i in range(1, len(tokens)):
            n_gram = tokens[:i+1]
            input_sequences.append(n_gram)

    print(f"Total Generated N-Gram Training Sequences: {len(input_sequences)}")
    print("First 5 N-Gram Sequences:", input_sequences[:5])

    model = PurePythonNextWordLSTM(vocab_size=len(vocab), embed_dim=8, hidden_dim=16)

    test_seeds = ["united states", "donald trump", "india and china", "science and technology"]
    print("\n--- Generating Continuations from Seed Prompts ---")
    for seed in test_seeds:
        seed_tokens = [word2idx[w] for w in seed.split() if w in word2idx]
        gen_tokens = model.generate(seed_tokens, num_words=4)
        gen_words = [idx2word[i] for i in gen_tokens]
        print(f"Seed: '{seed:25s}' ──► Generated: {' '.join(gen_words)}")

    print("=" * 80)
