# Module 01: Foundations of Generative AI & Recurrent Neural Networks (RNNs)

---

## 1. The Historical Context: The Road to Generative AI

Before the era of Large Language Models (LLMs) like GPT-4 and Claude, machine learning went through four major eras of text and sequence modeling:

```
[Rule-Based / N-Grams]  ──►  [Traditional Deep Learning (Feed-Forward)]
       (1980s-1990s)                            (2000s)
                                                  │
                                                  ▼
[Transformers & LLMs]   ◄──  [Sequential Models (RNNs, LSTMs, GRUs)]
      (2017-Present)                            (2010-2016)
```

1. **Rule-Based & Statistical $N$-Gram Models (1980s–1990s)**:
   - Models counted frequencies of word pairs or triplets ($P(w_3 \mid w_1, w_2)$).
   - **Flaw**: Combinatorial explosion; no concept of semantic similarity between words (e.g., "cat" and "kitten" were treated as completely unrelated tokens).

2. **Standard Feed-Forward Neural Networks (MLPs / ANN)**:
   - Allowed non-linear feature learning and dense vector embeddings (e.g., Word2Vec).
   - **Flaw**: Fixed-size input/output window; no inherent memory of word order.

3. **Recurrent Neural Networks (RNNs) & Gated Variants (LSTM/GRU)**:
   - Introduced recurrence and hidden memory states to process variable-length text sequentially.
   - **Flaw**: Vanishing gradients, sequential bottleneck (no GPU parallelization), and the "information compression" bottleneck.

4. **The Transformer Revolution (2017–Present)**:
   - Replaced recurrence with Self-Attention (*"Attention Is All You Need"*), allowing full sequence parallelization and direct token-to-token connections regardless of distance.

---

## 2. Core Concepts Demystified: Neurons, Layers, and Context

### A. What is a Neuron (Perceptron)?
At its core, an artificial neuron is a mathematical function that:
1. Takes a vector of numerical inputs $\mathbf{x} = [x_1, x_2, \dots, x_n]$.
2. Multiplies each input by a learnable **weight** $w_i$ (representing the importance of that input).
3. Adds a learnable **bias** $b$ (representing the baseline threshold/offset).
4. Passes the aggregate sum through a non-linear **activation function** $f(\cdot)$ (such as $\tanh$, $\text{ReLU}$, or $\text{Sigmoid}$).

$$\text{Output } z = f\left(\sum_{i=1}^n w_i x_i + b\right) = f(\mathbf{W}\mathbf{x} + b)$$

*Analogy*: Think of a neuron as a decision-maker weighing different pieces of evidence to produce a score.

---

### B. What is a Layer?
- A **layer** is a collection of parallel neurons operating simultaneously on the same input vector.
- **Input Layer**: Receives the raw numerical data (e.g., word embeddings).
- **Hidden Layers**: Intermediate transformations where the network learns hierarchical abstractions (e.g., syntax, grammar, tone).
- **Output Layer**: Produces the final task output (e.g., class probabilities for the next word).

---

### C. What is "Context"?
In Natural Language Processing (NLP), **context** refers to the surrounding words and historical information needed to understand the current word.

*Example*:
- *"The bank of the river was muddy."* $\rightarrow$ "bank" = land alongside water.
- *"I deposited money in the bank."* $\rightarrow$ "bank" = financial institution.

A standard Feed-Forward network treats every input independently. Without a memory mechanism, it cannot maintain **context** across time steps.

---

## 3. How Standard Neural Networks Fail on Sequences

Suppose we want to predict the next word in the sentence:
> *"The chef prepared an exquisite meal and served ___"* (Target: *it* / *them* / *the guests*)

A standard Multi-Layer Perceptron (MLP):
1. Requires a **fixed input size** (e.g., exactly 5 words). If a sentence is 20 words, it must be chopped or padded.
2. Does **not share weights across positions**. If "chef" is at position 1 vs. position 4, the network treats it with completely different weights.
3. Has **no temporal memory**: Order is lost unless manually engineered.

---

## 4. How Recurrent Neural Networks (RNNs) Work

The fundamental innovation of an RNN is the **recurrent feedback loop**. Instead of only accepting the current input $\mathbf{x}_t$, the network also accepts its own output from the previous time step, called the **Hidden State** ($\mathbf{h}_{t-1}$).

```
          Output y_t
              ▲
              │ W_hy
         ┌─────────┐
h_{t-1} ─►│ Hidden  │──► h_t (Passed to next step)
         │ State   │
         └─────────┘
              ▲
              │ W_xh
          Input x_t
```

### The Unrolled View Across Time

When processing a sequence of tokens $x_1, x_2, \dots, x_T$:

```
y_1                  y_2                  y_3                  y_T
 ▲                    ▲                    ▲                    ▲
 │ W_hy               │ W_hy               │ W_hy               │ W_hy
┌───┐   h_1          ┌───┐   h_2          ┌───┐        h_{T-1} ┌───┐   h_T
│RNN│ ──────►        │RNN│ ──────►        │RNN│ ──────► ... ──►│RNN│
└───┘   W_hh         └───┘   W_hh         └───┘        W_hh    └───┘
  ▲                    ▲                    ▲                    ▲
  │ W_xh               │ W_xh               │ W_xh               │ W_xh
 x_1                  x_2                  x_3                  x_T
(The)               (chef)              (cooked)              (food)
```

---

### Mathematical Equations

At each time step $t$:

1. **Compute New Hidden State ($h_t$)**:
   $$\mathbf{h}_t = \tanh\left(\mathbf{W}_{xh}\mathbf{x}_t + \mathbf{W}_{hh}\mathbf{h}_{t-1} + \mathbf{b}_h\right)$$
   - $\mathbf{x}_t \in \mathbb{R}^d$: Input embedding at step $t$.
   - $\mathbf{h}_{t-1} \in \mathbb{R}^h$: Hidden state from previous step ($t-1$).
   - $\mathbf{W}_{xh} \in \mathbb{R}^{h \times d}$: Weight matrix projecting current input to hidden dimension.
   - $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$: Weight matrix projecting previous memory into current hidden state.
   - $\mathbf{b}_h \in \mathbb{R}^h$: Hidden bias vector.
   - $\tanh$: Squashes values between $-1$ and $+1$, regulating the magnitude of activation.

2. **Compute Output Prediction ($y_t$)**:
   $$\mathbf{\hat{y}}_t = \text{Softmax}\left(\mathbf{W}_{hy}\mathbf{h}_t + \mathbf{b}_y\right)$$
   - $\mathbf{W}_{hy} \in \mathbb{R}^{V \times h}$: Weight matrix projecting hidden state to vocabulary size $V$.
   - $\text{Softmax}$: Produces a probability distribution over the next possible words.

---

## 5. Why Vanilla RNNs Failed: The Core Bottlenecks

Despite their theoretical beauty, vanilla RNNs suffered from severe structural limitations:

### 1. The "Information Compression" / Bottleneck Pressure on Last Hidden Layer
- In an encoder-decoder or sequence-classification RNN, the final hidden state $\mathbf{h}_T$ is expected to summarize the **entire meaning** of a 50-word or 500-word paragraph into a fixed vector (e.g., 256 or 512 numbers).
- As new inputs are processed, each multiplication by $\mathbf{W}_{hh}$ overwrites or dilutes earlier information.
- By the time the RNN reaches step 100, information from step 1 is virtually erased (**Catastrophic Forgetting**).

### 2. Vanishing and Exploding Gradients (Backpropagation Through Time - BPTT)
During training, gradients are computed via the chain rule backwards through every single time step:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{k=2}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{k=2}^T \mathbf{W}_{hh}^T \cdot \text{diag}(1 - \tanh^2(\dots))$$

- **Vanishing Gradients**: Because the derivative of $\tanh$ is between $0$ and $1$, and $\mathbf{W}_{hh}$ has eigenvalues $< 1$, multiplying these terms $T$ times causes the gradient to decay exponentially toward $0$. The network cannot learn long-range dependencies.
- **Exploding Gradients**: If the largest eigenvalue of $\mathbf{W}_{hh} > 1$, the gradient grows exponentially, causing numeric instability (NaNs).

### 3. Sequential Execution Bottleneck (GPU Unfriendliness)
- Step $t$ **strictly depends** on step $t-1$.
- Because of this recurrence relation, computations cannot be parallelized across the time dimension on GPUs. Training on large corpora took days to weeks for relatively small datasets.

---

## 6. Summary Comparison: Evolution of Sequence Models

| Feature | Vanilla RNN | LSTM / GRU | Transformer (LLMs) |
| :--- | :--- | :--- | :--- |
| **Memory Mechanism** | Single hidden state $h_t$ | Gating cells ($c_t, h_t$, forget gate) | Self-Attention Matrix ($Q, K, V$) |
| **Max Effective Context** | ~10–20 tokens | ~100–300 tokens | 128k–1M+ tokens |
| **GPU Parallelization** | No (Sequential) | No (Sequential) | **Yes (Fully Parallel)** |
| **Long-Range Gradients** | Vanishes exponentially | Additive cell highway | Direct $O(1)$ token-to-token paths |
