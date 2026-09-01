# Lecture: Build RNN from Scratch Theory
## End-to-End English-to-Hindi Seq2Seq Translation, Encoder-Decoder Mechanics, Matrix Computations, Softmax Deep Dive, and the Path to Modern GenAI

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

## 2. Vocabulary, Encodings & Matrix Dimensions

To keep the matrix arithmetic 100% concrete, we define a toy vocabulary with exact dimensional shapes.

### A. Vocabularies

- **English (Source Vocab $V_{src} = 5$)**:
  `{"<PAD>": 0, "You": 1, "are": 2, "good": 3, "<EOS>": 4}`
- **Hindi (Target Vocab $V_{tgt} = 6$)**:
  `{"<PAD>": 0, "<SOS>": 1, "आप": 2, "अच्छे": 3, "हैं": 4, "<EOS>": 5}`

### B. Dimensional Definitions
Let:
- $d_{in} = 4$: Dimension of token embeddings ($\mathbf{x}_t \in \mathbb{R}^{4 \times 1}$).
- $d_h = 3$: Dimension of hidden memory state ($\mathbf{h}_t \in \mathbb{R}^{3 \times 1}$).
- $V_{tgt} = 6$: Size of target vocabulary.

### C. Parameter Initialization (Random Weights & Biases)

#### 1. Encoder Parameters:
- $\mathbf{W}_{xh}^{enc} \in \mathbb{R}^{3 \times 4}$: Maps English input embedding to encoder hidden state.
- $\mathbf{W}_{hh}^{enc} \in \mathbb{R}^{3 \times 3}$: Maps previous encoder hidden state to current state.
- $\mathbf{b}_h^{enc} \in \mathbb{R}^{3 \times 1}$: Encoder hidden bias.

#### 2. Decoder Parameters:
- $\mathbf{W}_{ys}^{dec} \in \mathbb{R}^{3 \times 4}$: Maps previous target token embedding to decoder hidden state $\mathbf{s}_t$.
- $\mathbf{W}_{ss}^{dec} \in \mathbb{R}^{3 \times 3}$: Maps previous decoder state $\mathbf{s}_{t-1}$ to current state $\mathbf{s}_t$.
- $\mathbf{b}_s^{dec} \in \mathbb{R}^{3 \times 1}$: Decoder hidden bias.
- $\mathbf{W}_{vocab}^{dec} \in \mathbb{R}^{6 \times 3}$: Projects decoder state $\mathbf{s}_t$ to target vocabulary logits ($V_{tgt} \times d_h$).
- $\mathbf{b}_{vocab}^{dec} \in \mathbb{R}^{6 \times 1}$: Output vocabulary bias vector.

---

## 3. Step-by-Step Encoder Computations

The **Encoder** reads the input sentence token by token and compresses the entire sentence into the final hidden state $\mathbf{h}_3^{enc}$, known as the **Context Vector ($\mathbf{c}$)**.

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

## 4. Step-by-Step Decoder Computations

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

## 5. Softmax Function: Deep Mathematical Exploration

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

## 6. How RNNs Paved the Path to Attention & Transformers (2014–2017)

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

## 7. Complete Summary Table

| Concept | Vanilla Seq2Seq RNN | Attention-Augmented RNN | Modern Transformer (GPT/LLM) |
| :--- | :--- | :--- | :--- |
| **Context Representation** | Single fixed vector $\mathbf{h}_{enc\_T}$ | Dynamic weighted sum $\sum \alpha_i \mathbf{h}_i$ | Multi-Head Self-Attention Matrix $\mathbf{Q}\mathbf{K}^T$ |
| **Token-to-Token Path Length** | $O(T)$ sequential steps | $O(1)$ direct attention lookup | $O(1)$ parallel cross/self-attention |
| **Training Parallelization** | Sequential ($O(T)$ on GPU) | Sequential ($O(T)$ on GPU) | **Fully Parallel ($O(1)$ step)** |
| **Softmax Usage** | Final output vocabulary logits | Attention weights + Output logits | Self-Attention weights + Output logits |
