# Lecture 04: Code an RNN from Scratch
## Tiny Encoder-Decoder Implementation, Training Loops, Gradient Descent, Optimization Landscapes, RNN Failure Modes, and the Evolution to LSTMs

---

## 1. System Architecture: The Tiny Seq2Seq Translator

In this lecture, we implement a full end-to-end Neural Machine Translation pipeline from scratch using PyTorch parameters, translating:

$$\text{Source (English): } \text{"I go <EOS>"} \quad \xrightarrow{\text{Seq2Seq RNN}} \quad \text{Target (Hindi): } \text{"मैं जाता हूँ <EOS>"}$$

```
========================= TINY ENCODER =========================       ========================= TINY DECODER =========================
   "I" (0)            "go" (1)          "<EOS>" (2)                   <GO> (0)           "मैं" (1)         "जाता" (2)          "हूँ" (3)
      │                  │                  │                            │                  │                  │                  │
      ▼                  ▼                  ▼                            ▼                  ▼                  ▼                  ▼
[Embedding(0)]     [Embedding(1)]     [Embedding(2)]               [Embedding(0)]     [Embedding(1)]     [Embedding(2)]     [Embedding(3)]
      │                  │                  │                            │                  │                  │                  │
      ▼                  ▼                  ▼                            ▼                  ▼                  ▼                  ▼
┌───────────┐      ┌───────────┐      ┌───────────┐                ┌───────────┐      ┌───────────┐      ┌───────────┐      ┌───────────┐
│ TinyEnc   │─────►│ TinyEnc   │─────►│ TinyEnc   │ ──Context(h)──►│ TinyDec   │─────►│ TinyDec   │─────►│ TinyDec   │─────►│ TinyDec   │
└───────────┘  h_1 └───────────┘  h_2 └───────────┘    (shape: 2)  └───────────┘  s_1 └───────────┘  s_2 └───────────┘  s_3 └───────────┘  s_4
                                                                         │                  │                  │                  │
                                                                         ▼                  ▼                  ▼                  ▼
                                                                     [W_out @ s]        [W_out @ s]        [W_out @ s]        [W_out @ s]
                                                                         │                  │                  │                  │
                                                                         ▼                  ▼                  ▼                  ▼
                                                                     Logits_1           Logits_2           Logits_3           Logits_4
                                                                      (1 x 5)            (1 x 5)            (1 x 5)            (1 x 5)
                                                                         │                  │                  │                  │
                                                                  [Loss vs "मैं"]    [Loss vs "जाता"]   [Loss vs "हूँ"]    [Loss vs "<EOS>"]
```

---

## 2. Parameter Matrix Dimensions & Mathematical Formulas

Let:
- Source vocabulary size: $V_{src} = 3$ (`{"I": 0, "go": 1, "<EOS>": 2}`)
- Target vocabulary size: $V_{tgt} = 5$ (`{"<GO>": 0, "मैं": 1, "जाता": 2, "हूँ": 3, "<EOS>": 4}`)
- Embedding dimension: $d_{embed} = 1$
- Hidden state dimension: $d_{hidden} = 2$

### A. TinyEncoder Formulations
At each time step $t \in \{0, 1, 2\}$:
1. **Embedding Lookup**: $\mathbf{x}_t = \text{Embedding}(\text{token}_t) \in \mathbb{R}^1$
2. **Recurrent State Update**:
   $$\mathbf{h}_t = \tanh\left(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}\right)$$
   - $\mathbf{W}_h \in \mathbb{R}^{2 \times 2}$: Hidden-to-hidden recurrence matrix.
   - $\mathbf{W}_x \in \mathbb{R}^{2 \times 1}$: Input-to-hidden projection matrix.
   - $\mathbf{b} \in \mathbb{R}^2$: Hidden bias vector.
   - Initial state: $\mathbf{h}_{-1} = [0.0, 0.0]^T$.
   - Final state output: $\mathbf{h}_{final} = \mathbf{h}_2 \in \mathbb{R}^2$ (The Context Vector).

---

### B. TinyDecoder Formulations
At each decoding step $t \in \{0, 1, 2, 3\}$:
1. **Embedding Lookup**: $\mathbf{x}_t^{dec} = \text{Embedding}(\text{token}_t) \in \mathbb{R}^1$
2. **Hidden State Update**:
   $$\mathbf{h}_t^{dec} = \tanh\left(\mathbf{W}_h \mathbf{h}_{t-1}^{dec} + \mathbf{W}_x \mathbf{x}_t^{dec} + \mathbf{b}\right)$$
   - Initial state: $\mathbf{h}_{-1}^{dec} = \mathbf{h}_{final}^{enc} \in \mathbb{R}^2$.
3. **Logits Projection**:
   $$\mathbf{z}_t = \mathbf{W}_{out} \mathbf{h}_t^{dec} + \mathbf{b}_{out} \in \mathbb{R}^5$$
   - $\mathbf{W}_{out} \in \mathbb{R}^{5 \times 2}$: Hidden-to-vocabulary projection matrix.
   - $\mathbf{b}_{out} \in \mathbb{R}^5$: Output vocabulary bias vector.

---

## 3. Training Dynamics: Teacher Forcing, Loss & Optimization

### A. What is Teacher Forcing?
During training, the decoder is fed the **ground-truth target sequence** as input at each time step rather than its own generated prediction:

$$\text{Decoder Inputs: } [\text{"<GO>"}, \text{"मैं"}, \text{"जाता"}, \text{"हूँ"}] \quad \xrightarrow{\text{Predicts}} \quad \text{Targets: } [\text{"मैं"}, \text{"जाता"}, \text{"हूँ"}, \text{"<EOS>"}]$$

- **Benefit**: Prevents compounding error propagation early in training.
- **Exposure Bias (Inference Difference)**: During inference, ground-truth is absent; the model runs **autoregressively**, feeding its own previous prediction $\hat{y}_{t-1}$ as the input to step $t$.

---

### B. Loss Computation: Categorical Cross-Entropy
For each time step $t$, the loss measures the negative log-likelihood of the correct target token:

$$\mathcal{L} = \frac{1}{T} \sum_{t=1}^T -\log P(\text{target}_t) = \frac{1}{T} \sum_{t=1}^T \left( -z_{t,\text{target}} + \log \sum_{j=1}^{V_{tgt}} e^{z_{t,j}} \right)$$

PyTorch's `nn.CrossEntropyLoss()` combines `LogSoftmax` and `NLLLoss` into a single numerically stable operation.

---

### C. Gradient Descent & Parameter Updates

```
Forward Pass ──► Compute Loss L ──► Backward Pass (BPTT) ──► Gradient Descent Step
     (z)             (Loss)              (dL/dθ)             (θ = θ - η · dL/dθ)
```

1. `optimizer.zero_grad()`: Clears accumulated gradients from previous iteration.
2. `loss.backward()`: Computes $\nabla_\theta \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta}$ for all parameters via Backpropagation Through Time (BPTT).
3. `optimizer.step()`: Updates weights along the steepest descent direction:
   $$\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}$$
   where $\eta$ is the learning rate.

---

## 4. Optimization Landscapes: Local Minima, Maxima & Saddle Points

Neural network loss functions are highly **non-convex** high-dimensional surfaces:

```
        Loss L(θ)
          ▲
          │          Local Maxima (∇L = 0, ∇²L < 0)
          │               .-"-.
          │             .'     '.
          │   .-"-.    /         \     Saddle Point (∇L = 0, mixed curvature)
          │  /     \  /           \       _..._
          │ /       \/             \__.-'"     "'-.__
          │/   Local Minima                          \
          │  (∇L = 0, ∇²L > 0)                        \      Global Minimum
          │                                            \       (Lowest Loss)
          │                                             \___.-'
          └─────────────────────────────────────────────────────────────► Parameter θ
```

1. **Local Minima**: A point where the loss is lower than nearby points ($\nabla \mathcal{L} = 0$, Hessian $\nabla^2 \mathcal{L} \succ 0$), but not necessarily the global lowest error.
2. **Local Maxima**: Peaks where $\nabla \mathcal{L} = 0$, but Hessian is negative definite ($\nabla^2 \mathcal{L} \prec 0$). Gradient descent naturally moves away from maxima.
3. **Saddle Points**: Points where the gradient is zero ($\nabla \mathcal{L} = 0$), but curves upwards in some dimensions and downwards in others. In modern high-dimensional deep learning, **saddle points and plateaus** are far more common obstacles than local minima.
4. **Plateaus**: Flat regions with tiny gradients ($\nabla \mathcal{L} \approx 0$), causing training to stall if learning rate is too small.

---

## 5. Core Failure Modes of Vanilla RNNs

### A. Context Retention Bottleneck (Catastrophic Forgetting)
In our tiny model, all semantic meaning of `"I go <EOS>"` must be squeezed into just **$d_{hidden} = 2$ numbers**:

$$\mathbf{h}_{final} = [h_1, h_2]^T$$

For a 20-word sentence, earlier tokens undergo 20 consecutive matrix multiplications by $\mathbf{W}_h$ and $\tanh$ squashing. The distinctive signal of the first word is overwritten by later words.

---

### B. The Vanishing Gradient Problem (Backpropagation Through Time)
During backpropagation, the gradient of the loss with respect to the initial hidden state $\mathbf{h}_0$ expands via the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{h}_0} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{k=1}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{k=1}^T \mathbf{W}_h^T \cdot \text{diag}\left(1 - \tanh^2(\mathbf{z}_k)\right)$$

#### Why Gradients Disappear:
1. The derivative of $\tanh$ is bounded: $0 \le (1 - \tanh^2(z)) \le 1.0$.
2. If the spectral radius (eigenvalues) of $\mathbf{W}_h < 1$, multiplying $T$ matrices together causes the gradient to decay **exponentially toward zero**:
   $$\left\|\frac{\partial \mathcal{L}}{\partial \mathbf{h}_0}\right\| \propto (\lambda_{max})^T \xrightarrow{T \to \infty} 0$$
3. As a result, the weights $\mathbf{W}_x, \mathbf{W}_h$ receive zero gradient updates from distant past errors, making it impossible for the network to learn long-range grammar or dependencies.

---

## 6. How RNN Failures Paved the Path to LSTMs (Long Short-Term Memory)

To fix the vanishing gradient problem, Sepp Hochreiter & Jürgen Schmidhuber (1997) introduced the **LSTM (Long Short-Term Memory)** network.

```
                    ┌─────────────────────────┐
                    │   LSTM CELL STATE (C_t) │  <── Constant Error Highway (Additive!)
                    │   C_t = f_t*C_{t-1} + i_t*\tilde{C}_t
                    └─────────────────────────┘
                                ▲
              ┌─────────────────┴─────────────────┐
              │                                   │
      ┌───────────────┐                   ┌───────────────┐
      │  FORGET GATE  │                   │  INPUT GATE   │
      │  f_t = σ(...) │                   │  i_t = σ(...) │
      │  (What to drop)                   │  (What to add)│
      └───────────────┘                   └───────────────┘
```

### The Architectural Breakthrough of LSTMs:

#### 1. The Cell State Highway ($C_t$):
Instead of passing memory purely through non-linear matrix multiplications ($\tanh(\mathbf{W}_h \mathbf{h}_{t-1})$), LSTMs introduce an explicit **Cell State ($C_t$)** that acts as an additive memory conveyor belt:

$$\mathbf{C}_t = \mathbf{f}_t \odot \mathbf{C}_{t-1} + \mathbf{i}_t \odot \mathbf{\tilde{C}}_t$$

Because the update is **linear and additive** ($\mathbf{C}_{t-1} + \dots$), gradients can flow back through time across hundreds of steps without decaying to zero (**Constant Error Carousel**).

---

#### 2. The Three Gating Mechanisms:

| Gate | Mathematical Formula | Function / Intuition |
| :--- | :--- | :--- |
| **Forget Gate ($\mathbf{f}_t$)** | $\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{U}_f \mathbf{h}_{t-1} + \mathbf{b}_f)$ | Outputs $0 \to 1$. Decides what information to **discard** from previous cell state (e.g., forget old subject when a new subject appears). |
| **Input Gate ($\mathbf{i}_t$)** | $\mathbf{i}_t = \sigma(\mathbf{W}_i \mathbf{x}_t + \mathbf{U}_i \mathbf{h}_{t-1} + \mathbf{b}_i)$ | Decides **which values** of the new candidate memory to update. |
| **Candidate State ($\mathbf{\tilde{C}}_t$)** | $\mathbf{\tilde{C}}_t = \tanh(\mathbf{W}_c \mathbf{x}_t + \mathbf{U}_c \mathbf{h}_{t-1} + \mathbf{b}_c)$ | Creates a vector of new candidate concepts to add to the cell state. |
| **Output Gate ($\mathbf{o}_t$)** | $\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{U}_o \mathbf{h}_{t-1} + \mathbf{b}_o)$ | Decides what parts of the cell state $\mathbf{C}_t$ to output to hidden state $\mathbf{h}_t$. |
| **Hidden State ($\mathbf{h}_t$)** | $\mathbf{h}_t = \mathbf{o}_t \odot \tanh(\mathbf{C}_t)$ | The filtered output state passed to the next layer and next time step. |

---

## 7. Evolution Summary: From RNNs to Transformers

```
[Vanilla RNN (1986)]
  • Pure multiplicative state update: h_t = tanh(W_h h_{t-1} + W_x x_t)
  • Fatal Flaw: Vanishing gradients, short memory (<10 words)
          │
          ▼
[LSTM / GRU (1997 - 2014)]
  • Added Gating (Forget, Input, Output) + Additive Cell State (C_t)
  • Breakthrough: Solved vanishing gradients, extended memory to ~100 tokens
  • Remaining Flaw: Strictly sequential; cannot parallelize on GPUs
          │
          ▼
[Attention Mechanisms (Bahdanau 2014)]
  • Decoder dynamically attends to ALL encoder hidden states
  • Breakthrough: Eliminated fixed context vector bottleneck
          │
          ▼
[Transformers (Vaswani et al. 2017 - "Attention Is All You Need")]
  • Eliminated recurrence entirely; uses Self-Attention across all tokens
  • Modern Era: GPT-4, Claude, Gemini, LLaMA
```
