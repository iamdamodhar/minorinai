# Lecture: Build RNN from Scratch Theory
## End-to-End English-to-Hindi Seq2Seq Translation, Encoder-Decoder Mechanics, Matrix Computations, Softmax Deep Dive, Activation Dynamics (Why $\tanh$), and the Path to Modern GenAI

---

## 1. Introduction & Overview

In this lecture, we bridge the gap between abstract neural network theory and concrete, step-by-step matrix computations by building an **Encoder-Decoder Recurrent Neural Network (Seq2Seq)** from scratch for a machine translation task:

$$\text{Source (English): } \text{"You are good"} \quad \xrightarrow{\text{Seq2Seq RNN}} \quad \text{Target (Hindi): } \text{"आप अच्छे हैं"}$$

```
========================= ENCODER =========================       ========================= DECODER =========================
   "You"              "are"              "good"                     <SOS>              "आप"             "अच्छे"
     │                  │                  │                          │                  │                  │
     ▼                  ▼                  ▼                          ▼                  ▼                  ▼
┌─────────┐        ┌─────────┐        ┌─────────┐                ┌─────────┐        ┌─────────┐        ┌─────────┐
│ Enc-RNN │───────►│ Enc-RNN │───────►│ Enc-RNN │ ──Context Vector──►│ Dec-RNN │───────►│ Dec-RNN │───────►│ Dec-RNN │
└─────────┘  h_1   └─────────┘  h_2   └─────────┘   (h_enc_3)    └─────────┘  s_1   └─────────┘  s_2   └─────────┘  s_3
     ▲                  ▲                  ▲                          │                  │                  │
    x_1                x_2                x_3                         ▼                  ▼                  ▼
                                                                  [Softmax]          [Softmax]          [Softmax]
                                                                      │                  │                  │
                                                                      ▼                  ▼                  ▼
                                                                     "आप"             "अच्छे"             "हैं"
```

---

## 2. Vocabulary, Embeddings & Dimensional Arithmetic

### A. What is a Vocabulary ($V$)?
Computers cannot directly process human characters or words; they only perform arithmetic on numbers. A **Vocabulary** is an indexed dictionary mapping every unique recognized token to an integer ID:

$$\text{Vocab } \mathcal{V} = \{w_0: 0, w_1: 1, \dots, w_{|\mathcal{V}|-1}: |\mathcal{V}|-1\}$$

#### Special Tokens in Seq2Seq & LLMs:
- `<PAD>` (Index 0): Padding token to make batches of variable-length sentences equal in length.
- `<UNK>`: Unknown token for words not present in the training vocabulary.
- `<SOS>` / `<BOS>`: Start of Sentence/Sequence token; signals the Decoder to start generating.
- `<EOS>`: End of Sentence/Sequence token; signals the model that generation is finished.

#### Toy Vocabularies for Our Example:
- **English ($V_{src} = 5$)**: `{"<PAD>": 0, "You": 1, "are": 2, "good": 3, "<EOS>": 4}`
- **Hindi ($V_{tgt} = 6$)**: `{"<PAD>": 0, "<SOS>": 1, "आप": 2, "अच्छे": 3, "हैं": 4, "<EOS>": 5}`

---

### B. Why One-Hot Encoding Fails & The Need for Embeddings

If vocabulary size $V = 50,000$, one-hot encoding represents the word "You" as a 50,000-dimensional vector of zeros with a single `1` at index 1:

$$\mathbf{x}_{\text{"You"}} = [0, 1, 0, 0, \dots, 0]^T \in \mathbb{R}^{50,000}$$

#### Fatal Flaws of One-Hot Vectors:
1. **Severe Sparsity & Memory Inefficiency**: 99.998% of values are wasted zeros.
2. **Orthogonality & Zero Semantic Meaning**:
   $$\mathbf{x}_{\text{"king"}} \cdot \mathbf{x}_{\text{"queen"}} = 0, \quad \mathbf{x}_{\text{"apple"}} \cdot \mathbf{x}_{\text{"queen"}} = 0$$
   Every pair of words is equidistant and orthogonal in one-hot space. The network cannot tell that "king" and "queen" share royal/human attributes while "apple" is a fruit.

---

### C. Dense Continuous Word Embeddings ($\mathbf{E}$)

An **Embedding Matrix** $\mathbf{E} \in \mathbb{R}^{V \times d_{emb}}$ projects discrete token IDs into a dense, low-dimensional continuous vector space ($d_{emb} \ll V$, e.g., $d_{emb} = 4$ in our toy model, or $d_{emb} = 4096$ in LLaMA/GPT-4).

```
Embedding Matrix E (V x d_emb):
                Dim 1   Dim 2   Dim 3   Dim 4
[0] <PAD>    [  0.00,   0.00,   0.00,   0.00 ]
[1] "You"    [  0.82,  -0.34,   0.91,  -0.12 ]  <── Row 1 extracted
[2] "are"    [ -0.45,   0.78,  -0.15,   0.63 ]
[3] "good"   [  0.11,   0.95,   0.88,   0.42 ]
[4] "<EOS>"  [ -0.90,  -0.10,   0.05,  -0.80 ]
```

#### Key Geometric Properties of Learned Embeddings:
1. **Semantic Clustering**: Words with similar meanings have high cosine similarity:
   $$\text{Cosine Similarity}(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|} \approx 1.0$$
2. **Vector Arithmetic & Analogies**:
   $$\vec{v}_{\text{King}} - \vec{v}_{\text{Man}} + \vec{v}_{\text{Woman}} \approx \vec{v}_{\text{Queen}}$$

---

### D. Complete Dimensional Arithmetic Trace

Let:
- Input embedding dimension: $d_{in} = 4$
- Hidden memory dimension: $d_h = 3$
- Target vocabulary size: $V_{tgt} = 6$

```
[Token ID: 1] ──► [Embedding Matrix E (5x4)] ──► [x_t: (4x1)]
                                                       │
               ┌───────────────────────────────────────┘
               ▼
[W_xh: (3x4)] @ [x_t: (4x1)]  ──► (3x1) ──┐
[W_hh: (3x3)] @ [h_{t-1}: (3x1)] ──► (3x1) ──┼──► [Sum + b_h: (3x1)] ──► [tanh] ──► [h_t: (3x1)]
                                          │
               ┌──────────────────────────┘
               ▼
[W_vocab: (6x3)] @ [s_t: (3x1)] + [b_vocab: (6x1)] ──► [Logits z_t: (6x1)] ──► [Softmax] ──► [Probs P_t: (6x1)]
```

---

## 3. Why $\tanh$ is the Chosen Activation in RNNs

In standard deep learning, $\text{ReLU}$ is ubiquitous. However, for Vanilla RNNs and LSTM memory cells, **hyperbolic tangent ($\tanh$)** is the foundational activation function:

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}} \in (-1, +1)$$

```
        tanh(z)
          ▲  +1.0 ────────────────────── (Upper bound)
          │             .-------'
          │           .'
          │         .'
──────────┼────────(0,0)────────────────► z
          │       .'
          │     .'
          │   .'
          │.-'
          ▼  -1.0 ────────────────────── (Lower bound)
```

### 1. Bounded State Regulation (Preventing Value Explosion)
In an RNN, the hidden state is multiplied by $\mathbf{W}_{hh}$ repeatedly at every time step:

$$\mathbf{h}_t \approx \mathbf{W}_{hh} \cdot \mathbf{W}_{hh} \dots \mathbf{W}_{hh} \mathbf{h}_0 = (\mathbf{W}_{hh})^t \mathbf{h}_0$$

- If we used an unbounded activation like **$\text{ReLU}(z) = \max(0, z)$**, activations would compound and blow up toward $+\infty$ after 10–20 time steps.
- $\tanh$ strictly squashes every hidden state into the bounded range $[-1.0, +1.0]$, keeping numeric values stable across long sequences.

---

### 2. Zero-Centered Property (Eliminates Gradient Bias & Zigzagging)
- $\text{Sigmoid}(\sigma(z)) \in (0, 1)$ is strictly positive.
- When all inputs to a neuron are strictly positive, the gradient with respect to the weights during backpropagation ($\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \delta \cdot \mathbf{x}^T$) inherits the exact same sign. All weights must either increase together or decrease together, leading to severe **zigzagging dynamics** during gradient descent.
- $\tanh$ has a mean of zero ($\text{range } [-1, +1]$), allowing gradients to update weights in both positive and negative directions freely.

---

### 3. Ability to Represent Opposing & Inhibitory Information
- Human language contains negation, contrast, and erasure (e.g., *"not good"*, *"bad"* vs *"good"*).
- Because $\tanh$ outputs negative numbers, negative values can actively **suppress, invert, or diminish** previously accumulated memory signals in the hidden vector. Sigmoid (always positive) cannot cleanly perform this subtractive gating.

---

### Activation Comparison in Recurrent Architectures

| Activation Function | Output Range | Zero-Centered? | Bounded? | Why / Why Not in RNN Hidden State |
| :--- | :---: | :---: | :---: | :--- |
| **$\tanh$** | $(-1, +1)$ | **Yes** | **Yes** | **Gold Standard for RNN/LSTM**: Bounds state growth while allowing negative memory inhibition. |
| **Sigmoid ($\sigma$)** | $(0, 1)$ | No | Yes | Poor for hidden states (all-positive bias); excellent for **Gates** (0 = closed/forget, 1 = open/pass). |
| **$\text{ReLU}$** | $[0, \infty)$ | No | **No (Unbounded)** | Dangerous in vanilla RNNs (explodes values exponentially over $T$ steps); used in feedforward MLPs and Transformers. |

---

## 4. Step-by-Step Encoder Computations

The **Encoder** reads the English sentence token by token and compresses the entire sentence into the final hidden state $\mathbf{h}_3^{enc}$, known as the **Context Vector ($\mathbf{c}$)**.

$$\mathbf{h}_t^{enc} = \tanh\left(\mathbf{W}_{xh}^{enc}\mathbf{x}_t + \mathbf{W}_{hh}^{enc}\mathbf{h}_{t-1}^{enc} + \mathbf{b}_h^{enc}\right)$$

### Initial State:
$$\mathbf{h}_0^{enc} = \begin{bmatrix} 0.0 \\ 0.0 \\ 0.0 \end{bmatrix}$$

### Step 1: Input $x_1 =$ "You"
1. Input vector $\mathbf{x}_1 \in \mathbb{R}^{4 \times 1}$.
2. Linear sum: $\mathbf{z}_1 = \mathbf{W}_{xh}^{enc}\mathbf{x}_1 + \mathbf{W}_{hh}^{enc}\mathbf{h}_0^{enc} + \mathbf{b}_h^{enc}$.
3. Activation: $\mathbf{h}_1^{enc} = \tanh(\mathbf{z}_1)$.

### Step 2: Input $x_2 =$ "are"
1. Input vector $\mathbf{x}_2 \in \mathbb{R}^{4 \times 1}$.
2. Linear sum: $\mathbf{z}_2 = \mathbf{W}_{xh}^{enc}\mathbf{x}_2 + \mathbf{W}_{hh}^{enc}\mathbf{h}_1^{enc} + \mathbf{b}_h^{enc}$.
3. Activation: $\mathbf{h}_2^{enc} = \tanh(\mathbf{z}_2)$.

### Step 3: Input $x_3 =$ "good"
1. Input vector $\mathbf{x}_3 \in \mathbb{R}^{4 \times 1}$.
2. Linear sum: $\mathbf{z}_3 = \mathbf{W}_{xh}^{enc}\mathbf{x}_3 + \mathbf{W}_{hh}^{enc}\mathbf{h}_2^{enc} + \mathbf{b}_h^{enc}$.
3. Activation: $\mathbf{h}_3^{enc} = \tanh(\mathbf{z}_3)$.

$$\mathbf{c} = \mathbf{h}_3^{enc} \quad \text{(The 3-dimensional Context Vector)}$$

---

## 5. Step-by-Step Decoder Computations

The **Decoder** is an autoregressive model. Its job is to generate the Hindi translation one word at a time until it outputs `<EOS>`.

### Hand-off from Encoder to Decoder:
The initial hidden state of the decoder $\mathbf{s}_0$ is initialized with the Context Vector:
$$\mathbf{s}_0 = \mathbf{c} = \mathbf{h}_3^{enc}$$

### Decoder State Update Equation:
$$\mathbf{s}_t = \tanh\left(\mathbf{W}_{ys}^{dec}\mathbf{y}_{t-1}^{emb} + \mathbf{W}_{ss}^{dec}\mathbf{s}_{t-1} + \mathbf{b}_s^{dec}\right)$$

### Logits (Score Vector before Softmax):
$$\mathbf{z}_t = \mathbf{W}_{vocab}^{dec}\mathbf{s}_t + \mathbf{b}_{vocab}^{dec} \quad (\in \mathbb{R}^{6 \times 1})$$

### Target Probabilities via Softmax:
$$\mathbf{P}_t = \text{Softmax}(\mathbf{z}_t) = \frac{\exp(\mathbf{z}_t)}{\sum_{j=1}^6 \exp(z_{t,j})}$$

$$\hat{y}_t = \arg\max(\mathbf{P}_t)$$

```
Time Step 1: Input = <SOS>  ──► Dec-RNN(s_0) ──► s_1 ──► Logits z_1 ──► Softmax ──► Predicts "आप"
Time Step 2: Input = "आप"   ──► Dec-RNN(s_1) ──► s_2 ──► Logits z_2 ──► Softmax ──► Predicts "अच्छे"
Time Step 3: Input = "अच्छे" ──► Dec-RNN(s_2) ──► s_3 ──► Logits z_3 ──► Softmax ──► Predicts "हैं"
Time Step 4: Input = "हैं"   ──► Dec-RNN(s_3) ──► s_4 ──► Logits z_4 ──► Softmax ──► Predicts "<EOS>" (Halt)
```

---

## 6. Softmax Function: Deep Mathematical Exploration

### A. Mathematical Definition
Given a raw unnormalized logit vector $\mathbf{z} = [z_1, z_2, \dots, z_V]^T \in \mathbb{R}^V$:

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^V e^{z_j}} \quad \text{for } i \in \{1, \dots, V\}$$

### B. Matrix Computation & Numerical Stability (The Max-Subtraction Trick)
In practice, if a logit $z_i = 1000$, computing $e^{1000}$ results in floating-point overflow (`inf`). 

To make it numerically stable, we subtract the maximum logit $z_{max} = \max_j(z_j)$ before exponentiating:

$$\text{Softmax}(z_i) = \frac{e^{z_i - z_{max}}}{\sum_{j=1}^V e^{z_j - z_{max}}}$$

*Proof*:
$$\frac{e^{z_i - C}}{\sum_j e^{z_j - C}} = \frac{e^{z_i} \cdot e^{-C}}{\sum_j (e^{z_j} \cdot e^{-C})} = \frac{e^{z_i} e^{-C}}{e^{-C} \sum_j e^{z_j}} = \frac{e^{z_i}}{\sum_j e^{z_j}}$$
Since the constant $e^{-C}$ cancels out, the mathematical result is identical, but the maximum exponent is $e^0 = 1$, preventing any numerical overflow.

---

### C. Merits & Demerits of Softmax

| Category | Description |
| :--- | :--- |
| **Merits** | 1. **Converts Real Numbers to Valid Probabilities**: All outputs $p_i \in (0, 1)$ and $\sum p_i = 1.0$.<br>2. **Smoothly Differentiable**: Enables gradient backpropagation during training.<br>3. **Non-Linear Winner Amplification**: Exponentiation widens the gap between the highest logit and lower logits. |
| **Demerits** | 1. **$O(V)$ Computational Bottleneck**: Computing the denominator $\sum_{j=1}^V e^{z_j}$ requires summing over all vocabulary tokens. When $V = 100,000$, this is computationally expensive.<br>2. **Over-Confidence**: Exponentiation can assign near 100% confidence to slightly higher logits, masking uncertainty.<br>3. **Sensitivity to Extreme Logits**: High magnitude differences can cause vanishing gradients in saturated regions. |

---

### D. Why Softmax Remains Critical in Modern GPT / LLM Architectures

Even though Transformers replaced recurrence, **Softmax is used in the two most important places in modern Generative AI**:

#### 1. The Core of Self-Attention (Query-Key Alignment):
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
- Softmax converts the dot-product similarity between query tokens and key tokens into attention weights that sum to $1.0$.

#### 2. The Language Model Output Head (Next-Token Generation):
$$\mathbf{P}(w_{t+1} \mid w_{1:t}) = \text{Softmax}\left(\frac{\mathbf{z}}{T}\right)$$
- **Temperature ($T$)**:
  - $T \to 0$ (Argmax / Greedy): Distribution becomes sharp (deterministic).
  - $T > 1$: Flattens the distribution (more creative / random sampling).
- Top-$k$ and Top-$p$ (Nucleus) sampling algorithms filter this exact Softmax probability distribution.

---

## 7. How RNNs Paved the Path to Attention & Transformers (2014–2017)

```
[1. Vanilla Seq2Seq RNN (2014)]
   • Sutskever et al. / Cho et al.
   • Entire sentence squeezed into 1 fixed Context Vector h_enc_T.
   • Bottleneck: Long sentences (>15 words) lost early context.
                │
                ▼
[2. Bahdanau Attention (2014 - "Additive Attention")]
   • Decoder no longer relies only on h_enc_T.
   • At every decode step t, the decoder computes dynamic attention weights
     over ALL encoder states [h_1, h_2, ..., h_T].
                │
                ▼
[3. Transformers: "Attention Is All You Need" (Vaswani et al., 2017)]
   • Realization: "If attention connects any token to any token directly,
     why are we still computing steps sequentially through RNN loops?"
   • Replaced RNN recurrent loops with Self-Attention + Positional Encoding.
   • Result: 100% GPU parallelization and modern LLMs (GPT, Gemini, Claude).
```

---

## 8. Complete Summary Comparison Table

| Concept | Vanilla Seq2Seq RNN | Attention-Augmented RNN | Modern Transformer (GPT/LLM) |
| :--- | :--- | :--- | :--- |
| **Context Representation** | Single fixed vector $\mathbf{h}_{enc\_T}$ | Dynamic weighted sum $\sum \alpha_i \mathbf{h}_i$ | Multi-Head Self-Attention Matrix $\mathbf{Q}\mathbf{K}^T$ |
| **Activation in State** | Hyperbolic Tangent ($\tanh$) | Hyperbolic Tangent ($\tanh$) | $\text{GeLU}$ / $\text{SwiGLU}$ in FFN layers |
| **Token-to-Token Path Length** | $O(T)$ sequential steps | $O(1)$ direct attention lookup | $O(1)$ parallel cross/self-attention |
| **Training Parallelization** | Sequential ($O(T)$ on GPU) | Sequential ($O(T)$ on GPU) | **Fully Parallel ($O(1)$ step)** |
| **Softmax Usage** | Final output vocabulary logits | Attention weights + Output logits | Self-Attention weights + Output logits |
