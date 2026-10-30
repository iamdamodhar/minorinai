# Handwritten Assignment Master Guide: LSTM for Next-Word Generation
### *Exact Page-by-Page Blueprint for Handwritten Submission (Scanned PDF)*

> **How to use this guide while writing on paper / iPad**:
> - This document is organized into **7 clean, self-contained pages** (Pages 1 to 7).
> - Copy each page's headings, box diagrams, matrices, and equations directly onto ruled or plain A4 sheets.
> - **Every single matrix dimension** (`(m × n)`) is explicitly written underneath every matrix and vector so you satisfy 100% of the grading rubric (*"Explicit Computation: No assumptions—every step must be written explicitly. Dimensions Required: Each matrix operation should have clear dimensions mentioned"*).
> - All intermediate calculations use **4-decimal-place scientific calculator precision** (verified to match exact floating-point math).

---

# 📄 PAGE 1: From Vanilla RNN Bottlenecks to the LSTM Solution

## 1.1 Vanilla RNN: What It Does & What Is Missing

In sequence tasks such as **Next-Word Generation**, a neural network must read a sequence of previous words $x_1, x_2, \dots, x_T$ and predict the next word $\hat{w}_{T+1}$.

A **Vanilla Recurrent Neural Network (RNN)** maintains a single hidden state vector $h_t$ updated at every time step $t$:

$$h_t = \tanh\left(W_{hh} h_{t-1} + W_{xh} x_t + b_h\right)$$

### Hand-Drawable Diagram 1: Vanilla RNN Cell (Single Memory Line)

```text
                       h_t  (Output to Logits / Next Step)
                        ▲
                        │
         ┌──────────────┴──────────────┐
         │                             │
  h_{t-1}│      ┌───────────────┐      │
 ────────┼─────►│  tanh(W_h·h   │──────┴──────► h_t
 (Short  │      │   + W_x·x + b)│             (Single Overwritten
  Memory)│      └───────▲───────┘              Memory Vector)
         │              │
         └──────────────┼──────────────┐
                        │              │
                       x_t        [ Vanilla RNN Cell ]
                 (Current Token)
```

### What Is Missing in Vanilla RNN? (The 2 Core Bottlenecks)

1. **Bottleneck 1 — Single Mutable Memory Stream (Catastrophic Overwriting)**:
   - Vanilla RNN forces **both** *immediate short-term syntax* and *long-term semantic context* into a **single vector $h_t$**.
   - At every token step $t$, the entire memory $h_{t-1}$ is multiplied by $W_{hh}$ and squashed through $\tanh(\cdot)$. There is **no mechanism to protect** an important early word from being overwritten by later words (*"Goldfish memory"*).

2. **Bottleneck 2 — The Vanishing Gradient Problem (Backpropagation Through Time)**:
   - During training, gradients flow backward from step $T$ to step $1$ via the Chain Rule:
     $$\frac{\partial \mathcal{L}_T}{\partial h_1} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=2}^{T} \frac{\partial h_k}{\partial h_{k-1}} = \frac{\partial \mathcal{L}_T}{\partial h_T} \prod_{k=2}^{T} W_{hh}^T \cdot \text{diag}\!\left(1 - \tanh^2(z_k)\right)$$
   - Since $0 < 1 - \tanh^2(z_k) \le 1$, repeatedly multiplying numbers smaller than $1$ across many steps drives $\frac{\partial \mathcal{L}_T}{\partial h_1} \longrightarrow 0$ exponentially fast. Early words receive zero gradient signal and cannot be learned.

---

## 1.2 How LSTM Solves the RNN Bottleneck

**Long Short-Term Memory (LSTM)** (Hochreiter & Schmidhuber, 1997) solves both bottlenecks via two structural innovations:

1. **Dual-Stream Memory Architecture**:
   - **Cell State ($C_t \in \mathbb{R}^{h \times 1}$)** — **Long-Term Memory Highway**: Runs horizontally along the top of the cell using **linear, element-wise addition** ($+$) rather than matrix multiplication + $\tanh$ squashing. Because $\frac{\partial C_t}{\partial C_{t-1}} = f_t$, when the forget gate $f_t \approx 1$, gradients flow backward unimpeded (**Constant Error Carousel**).
   - **Hidden State ($h_t \in \mathbb{R}^{h \times 1}$)** — **Short-Term Working Memory**: Exposes only the currently relevant slice of memory needed for immediate next-word prediction.

2. **Three Learnable Sigmoid Gates ($\sigma \in (0, 1)$)**:
   - Act like **continuous valves ($0\%$ = completely closed, $100\%$ = completely open)** to selectively **forget** stale facts ($f_t$), **write** new facts ($i_t$), and **read/output** relevant context ($o_t$).

---

# 📄 PAGE 2: LSTM Cell Internals & Unrolled Architecture ($c = 3$)

## 2.1 Inside a Single LSTM Cell (Draw This Diagram Clearly!)

```text
  Long-Term Cell State Highway (Constant Error Carousel)
  C_{t-1} ────────────►( × )───────────────────►( + )──────────────────────► C_t
  (2×1)                  ▲                        ▲               │         (2×1)
                         │ f_t                    │ i_t ⊙ C̃_t     ▼
                         │ (Forget)               │            [ tanh ]
                         │                      ( × )             │
                         │                     ▲     ▲            ▼
                         │                i_t  │     │ C̃_t      ( × )──────► h_t
                         │              (Input)│     │(Cand.)     ▲         (2×1)
                         │                     │     │            │ o_t
                      ┌──┴──┐               ┌──┴──┐┌─┴────┐    ┌──┴──┐     (Output)
                      │  σ  │               │  σ  ││ tanh │    │  σ  │
                      │ W_f │               │ W_i ││ W_c  │    │ W_o │
                      └──▲──┘               └──▲──┘└──▲───┘    └──▲──┘
                         │                     │      │           │
  h_{t-1} (2×1) ────►[Concatenate: z_t = [h_{t-1}, x_t] (4×1)]────┘
                         ▲
                         │
                     x_t (2×1)
                (Token Embedding)
```

### The 6 Governing Equations Inside the LSTM Cell
Let the concatenated input column vector be $z_t = [h_{t-1}, x_t] = \begin{bmatrix} h_{t-1} \\ x_t \end{bmatrix}$ of dimension $((h + e) \times 1) = (4 \times 1)$:

1. **Forget Gate** (*How much of $C_{t-1}$ to keep, $\in (0,1)$*):
   $$f_t = \sigma\left(W_f [h_{t-1}, x_t] + b_f\right) \qquad \text{Shape: } (2 \times 4)(4 \times 1) + (2 \times 1) = (2 \times 1)$$
2. **Input Gate** (*How much new candidate info to write, $\in (0,1)$*):
   $$i_t = \sigma\left(W_i [h_{t-1}, x_t] + b_i\right) \qquad \text{Shape: } (2 \times 4)(4 \times 1) + (2 \times 1) = (2 \times 1)$$
3. **Output Gate** (*How much of updated cell memory to expose as $h_t$, $\in (0,1)$*):
   $$o_t = \sigma\left(W_o [h_{t-1}, x_t] + b_o\right) \qquad \text{Shape: } (2 \times 4)(4 \times 1) + (2 \times 1) = (2 \times 1)$$
4. **Candidate Cell State** (*New proposed semantic content, $\in (-1, +1)$*):
   $$\tilde{C}_t = \tanh\left(W_c [h_{t-1}, x_t] + b_c\right) \qquad \text{Shape: } (2 \times 4)(4 \times 1) + (2 \times 1) = (2 \times 1)$$
5. **Cell State Update** (*Element-wise Hadamard product $\odot$ + addition*):
   $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t \qquad \text{Shape: } (2 \times 1) \odot (2 \times 1) + (2 \times 1) \odot (2 \times 1) = (2 \times 1)$$
6. **Hidden State Update** (*Bounded working memory output*):
   $$h_t = o_t \odot \tanh(C_t) \qquad \text{Shape: } (2 \times 1) \odot (2 \times 1) = (2 \times 1)$$

> **Activation Definitions**:
> $$\sigma(u) = \frac{1}{1 + e^{-u}} \in (0, 1), \qquad \tanh(u) = \frac{e^{u} - e^{-u}}{e^{u} + e^{-u}} \in (-1, +1)$$

---

## 2.2 Unrolled Next-Word Generator Schematic for Context Size $c = 3$

```text
  Initial States: h_0 = [0, 0]^T (2×1),  C_0 = [0, 0]^T (2×1)
                         │
                         ▼
  Token 1 (ID=0: "I")  ──► [x_1 (2×1)] ──► ┌───────────┐
                                           │ LSTM Cell │──► h_1 (2×1), C_1 (2×1)
                                           └───────────┘         │
                                                                 ▼
  Token 2 (ID=1: "love")─► [x_2 (2×1)] ──► ┌───────────┐
                                           │ LSTM Cell │──► h_2 (2×1), C_2 (2×1)
                                           └───────────┘         │
                                                                 ▼
  Token 3 (ID=2: "deep")─► [x_3 (2×1)] ──► ┌───────────┐
                                           │ LSTM Cell │──► h_3 (2×1) ──► [Dense Logits: y = W_out·h_3 + b_out (5×1)]
                                           └───────────┘                                    │
                                                                                            ▼
                                                                             [Softmax(y) (5×1)] ──► argmax = ID 3 ("learning")
```

---

# 📄 PAGE 3: Problem Setup, Embeddings & Explicit Parameters

## 3.1 Given Hyperparameters & Vocabulary
- **Vocabulary Size**: $V = 5$
- **Hidden State Size**: $h = 2$
- **Token Embedding Size**: $e = 2$
- **Context Size**: $c = 3$
- **Concatenated Input Dimension**: $h + e = 2 + 2 = 4$

**Vocabulary Mapping ($V = 5$)**:
| Token ID | Word | Semantic Role |
| :---: | :---: | :--- |
| `0` | `"I"` | Subject Pronoun |
| `1` | `"love"` | Action / Sentiment Verb |
| `2` | `"deep"` | Technical Modifier |
| `3` | `"learning"` | Target Technical Noun |
| `4` | `"<EOS>"` | End-of-Sentence Token |

**Input Context Sequence ($c = 3$ tokens)**:
$$\text{Context Words} = [\text{"I"}, \text{"love"}, \text{"deep"}] \implies \text{Context Token IDs} = [0, 1, 2]$$

---

## 3.2 Step 1: Token Embedding Layer ($E \in \mathbb{R}^{V \times e} = 5 \times 2$)

We define the Embedding Lookup Matrix $E$ of shape $(5 \times 2)$ (where Column 1 encodes *Subject/Syntax* and Column 2 encodes *Sentiment/Technical Topic*):

$$E = \begin{bmatrix}
1.0 & 0.0 \\
0.5 & 1.0 \\
0.0 & 1.0 \\
-0.5 & 1.0 \\
0.0 & -1.0
\end{bmatrix}_{(5 \times 2)}
\begin{array}{l}
\leftarrow \text{ID } 0 \text{ ("I")} \\
\leftarrow \text{ID } 1 \text{ ("love")} \\
\leftarrow \text{ID } 2 \text{ ("deep")} \\
\leftarrow \text{ID } 3 \text{ ("learning")} \\
\leftarrow \text{ID } 4 \text{ ("<EOS>")}
\end{array}$$

Extracting column embeddings $x_t = (E^T \cdot \text{one\_hot}(\text{ID}_t)) \in \mathbb{R}^{2 \times 1}$ for our 3 context tokens `[0, 1, 2]`:

$$x_1 = E^T \begin{bmatrix} 1 \\ 0 \\ 0 \\ 0 \\ 0 \end{bmatrix}_{(5 \times 1)} = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}, \qquad
x_2 = E^T \begin{bmatrix} 0 \\ 1 \\ 0 \\ 0 \\ 0 \end{bmatrix}_{(5 \times 1)} = \begin{bmatrix} 0.5 \\ 1.0 \end{bmatrix}_{(2 \times 1)}, \qquad
x_3 = E^T \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \\ 0 \end{bmatrix}_{(5 \times 1)} = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}_{(2 \times 1)}$$

---

## 3.3 LSTM & Logits Layer Parameters (With Explicit Shapes)

**Initial States ($t = 0$)**:
$$h_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}, \qquad C_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}$$

**LSTM Gate Weights $(h \times (h+e)) = (2 \times 4)$ and Biases $(h \times 1) = (2 \times 1)$**:
*(Note: Columns 1–2 multiply $h_{t-1}$; Columns 3–4 multiply $x_t$)*

1. **Forget Gate Parameters**:
   $$W_f = \begin{bmatrix} 0.5 & 0.0 & 0.5 & 0.2 \\ 0.0 & 0.5 & -0.2 & 0.5 \end{bmatrix}_{(2 \times 4)}, \qquad b_f = \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)}$$

2. **Input Gate Parameters**:
   $$W_i = \begin{bmatrix} 0.4 & -0.2 & 0.8 & 0.2 \\ -0.2 & 0.4 & 0.2 & 0.8 \end{bmatrix}_{(2 \times 4)}, \qquad b_i = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}$$

3. **Output Gate Parameters**:
   $$W_o = \begin{bmatrix} 0.5 & 0.2 & 0.6 & 0.2 \\ 0.2 & 0.5 & 0.2 & 0.6 \end{bmatrix}_{(2 \times 4)}, \qquad b_o = \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}_{(2 \times 1)}$$

4. **Candidate Cell Parameters**:
   $$W_c = \begin{bmatrix} 0.5 & 0.0 & 1.0 & -0.5 \\ 0.0 & 0.5 & -0.5 & 1.0 \end{bmatrix}_{(2 \times 4)}, \qquad b_c = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}$$

5. **Dense Logits Layer Parameters $(V \times h) = (5 \times 2)$ and $(V \times 1) = (5 \times 1)$**:
   $$W_{\text{out}} = \begin{bmatrix}
   0.5 & -1.0 \\
   0.8 & 0.2 \\
   -0.4 & 0.6 \\
   -0.5 & 2.0 \\
   -0.2 & -0.8
   \end{bmatrix}_{(5 \times 2)}, \qquad
   b_{\text{out}} = \begin{bmatrix}
   0.0 \\
   0.0 \\
   0.1 \\
   0.5 \\
   -0.2
   \end{bmatrix}_{(5 \times 1)}$$

---

# 📄 PAGE 4: Step 2A — LSTM Forward Pass at Time Step $t = 1$ (Token `0`: `"I"`)

### A. Concatenate Previous Hidden State $h_0$ and Current Embedding $x_1$
$$[h_0, x_1] = \begin{bmatrix} h_{0,1} \\ h_{0,2} \\ x_{1,1} \\ x_{1,2} \end{bmatrix}_{(4 \times 1)} = \begin{bmatrix} 0.0 \\ 0.0 \\ 1.0 \\ 0.0 \end{bmatrix}_{(4 \times 1)}$$

### B. Forget Gate $f_1 = \sigma(W_f [h_0, x_1] + b_f)$
$$W_f [h_0, x_1] + b_f = \begin{bmatrix} 0.5 & 0.0 & 0.5 & 0.2 \\ 0.0 & 0.5 & -0.2 & 0.5 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.0 \\ 0.0 \\ 1.0 \\ 0.0 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} (0.5)(0.0) + (0.0)(0.0) + (0.5)(1.0) + (0.2)(0.0) \\ (0.0)(0.0) + (0.5)(0.0) + (-0.2)(1.0) + (0.5)(0.0) \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.5 \\ -0.2 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 1.0000 \\ 0.3000 \end{bmatrix}_{(2 \times 1)}$$
$$f_1 = \sigma\!\left(\begin{bmatrix} 1.0000 \\ 0.3000 \end{bmatrix}\right) = \begin{bmatrix} \frac{1}{1 + e^{-1.0}} \\ \frac{1}{1 + e^{-0.3}} \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \frac{1}{1 + 0.3679} \\ \frac{1}{1 + 0.7408} \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.7311} \\ \mathbf{0.5744} \end{bmatrix}_{(2 \times 1)}$$

### C. Input Gate $i_1 = \sigma(W_i [h_0, x_1] + b_i)$
$$W_i [h_0, x_1] + b_i = \begin{bmatrix} 0.4 & -0.2 & 0.8 & 0.2 \\ -0.2 & 0.4 & 0.2 & 0.8 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.0 \\ 0.0 \\ 1.0 \\ 0.0 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} (0.4)(0.0) + (-0.2)(0.0) + (0.8)(1.0) + (0.2)(0.0) \\ (-0.2)(0.0) + (0.4)(0.0) + (0.2)(1.0) + (0.8)(0.0) \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.8000 \\ 0.2000 \end{bmatrix}_{(2 \times 1)}$$
$$i_1 = \sigma\!\left(\begin{bmatrix} 0.8000 \\ 0.2000 \end{bmatrix}\right) = \begin{bmatrix} \frac{1}{1 + e^{-0.8}} \\ \frac{1}{1 + e^{-0.2}} \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \frac{1}{1 + 0.4493} \\ \frac{1}{1 + 0.8187} \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.6900} \\ \mathbf{0.5498} \end{bmatrix}_{(2 \times 1)}$$

### D. Output Gate $o_1 = \sigma(W_o [h_0, x_1] + b_o)$
$$W_o [h_0, x_1] + b_o = \begin{bmatrix} 0.5 & 0.2 & 0.6 & 0.2 \\ 0.2 & 0.5 & 0.2 & 0.6 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.0 \\ 0.0 \\ 1.0 \\ 0.0 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} 0.6 \\ 0.2 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.7000 \\ 0.3000 \end{bmatrix}_{(2 \times 1)}$$
$$o_1 = \sigma\!\left(\begin{bmatrix} 0.7000 \\ 0.3000 \end{bmatrix}\right) = \begin{bmatrix} \frac{1}{1 + e^{-0.7}} \\ \frac{1}{1 + e^{-0.3}} \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \frac{1}{1 + 0.4966} \\ \frac{1}{1 + 0.7408} \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.6682} \\ \mathbf{0.5744} \end{bmatrix}_{(2 \times 1)}$$

### E. Candidate Cell State $\tilde{C}_1 = \tanh(W_c [h_0, x_1] + b_c)$
$$W_c [h_0, x_1] + b_c = \begin{bmatrix} 0.5 & 0.0 & 1.0 & -0.5 \\ 0.0 & 0.5 & -0.5 & 1.0 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.0 \\ 0.0 \\ 1.0 \\ 0.0 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 1.0000 \\ -0.5000 \end{bmatrix}_{(2 \times 1)}$$
$$\tilde{C}_1 = \tanh\!\left(\begin{bmatrix} 1.0000 \\ -0.5000 \end{bmatrix}\right) = \begin{bmatrix} \mathbf{0.7616} \\ \mathbf{-0.4621} \end{bmatrix}_{(2 \times 1)}$$

### F. Cell State $C_1$ & Hidden State $h_1$ Updates
$$C_1 = f_1 \odot C_0 + i_1 \odot \tilde{C}_1 = \begin{bmatrix} 0.7311 \\ 0.5744 \end{bmatrix}_{(2 \times 1)} \odot \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.6900 \\ 0.5498 \end{bmatrix}_{(2 \times 1)} \odot \begin{bmatrix} 0.7616 \\ -0.4621 \end{bmatrix}_{(2 \times 1)}$$
$$C_1 = \begin{bmatrix} 0.0000 \\ 0.0000 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} (0.6900)(0.7616) \\ (0.5498)(-0.4621) \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.5255} \\ \mathbf{-0.2541} \end{bmatrix}_{(2 \times 1)}$$

$$h_1 = o_1 \odot \tanh(C_1) = \begin{bmatrix} 0.6682 \\ 0.5744 \end{bmatrix}_{(2 \times 1)} \odot \tanh\!\left(\begin{bmatrix} 0.5255 \\ -0.2541 \end{bmatrix}\right) = \begin{bmatrix} 0.6682 \\ 0.5744 \end{bmatrix}_{(2 \times 1)} \odot \begin{bmatrix} 0.4819 \\ -0.2488 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.3220} \\ \mathbf{-0.1429} \end{bmatrix}_{(2 \times 1)}$$

---

# 📄 PAGE 5: Step 2B — LSTM Forward Pass at Time Step $t = 2$ (Token `1`: `"love"`)

### A. Concatenate Previous Hidden State $h_1$ and Current Embedding $x_2$
$$[h_1, x_2] = \begin{bmatrix} 0.3220 \\ -0.1429 \\ 0.5000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)}$$

### B. Forget Gate $f_2 = \sigma(W_f [h_1, x_2] + b_f)$
$$W_f [h_1, x_2] + b_f = \begin{bmatrix} 0.5 & 0.0 & 0.5 & 0.2 \\ 0.0 & 0.5 & -0.2 & 0.5 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.3220 \\ -0.1429 \\ 0.5000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} (0.5)(0.3220) + (0.0)(-0.1429) + (0.5)(0.5000) + (0.2)(1.0000) \\ (0.0)(0.3220) + (0.5)(-0.1429) + (-0.2)(0.5000) + (0.5)(1.0000) \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} 0.1610 + 0.0000 + 0.2500 + 0.2000 \\ 0.0000 - 0.07145 - 0.1000 + 0.5000 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.6110 \\ 0.3286 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 1.1110 \\ 0.8286 \end{bmatrix}_{(2 \times 1)}$$
$$f_2 = \sigma\!\left(\begin{bmatrix} 1.1110 \\ 0.8286 \end{bmatrix}\right) = \begin{bmatrix} \frac{1}{1 + e^{-1.1110}} \\ \frac{1}{1 + e^{-0.8286}} \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.7523} \\ \mathbf{0.6961} \end{bmatrix}_{(2 \times 1)}$$

### C. Input Gate $i_2 = \sigma(W_i [h_1, x_2] + b_i)$
$$W_i [h_1, x_2] + b_i = \begin{bmatrix} 0.4 & -0.2 & 0.8 & 0.2 \\ -0.2 & 0.4 & 0.2 & 0.8 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.3220 \\ -0.1429 \\ 0.5000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} (0.4)(0.3220) + (-0.2)(-0.1429) + (0.8)(0.5000) + (0.2)(1.0000) \\ (-0.2)(0.3220) + (0.4)(-0.1429) + (0.2)(0.5000) + (0.8)(1.0000) \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} 0.1288 + 0.02858 + 0.4000 + 0.2000 \\ -0.0644 - 0.05716 + 0.1000 + 0.8000 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.7574 \\ 0.7784 \end{bmatrix}_{(2 \times 1)}$$
$$i_2 = \sigma\!\left(\begin{bmatrix} 0.7574 \\ 0.7784 \end{bmatrix}\right) = \begin{bmatrix} \mathbf{0.6808} \\ \mathbf{0.6853} \end{bmatrix}_{(2 \times 1)}$$

### D. Output Gate $o_2 = \sigma(W_o [h_1, x_2] + b_o)$
$$W_o [h_1, x_2] + b_o = \begin{bmatrix} 0.5 & 0.2 & 0.6 & 0.2 \\ 0.2 & 0.5 & 0.2 & 0.6 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.3220 \\ -0.1429 \\ 0.5000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} 0.1610 - 0.02858 + 0.3000 + 0.2000 \\ 0.0644 - 0.07145 + 0.1000 + 0.6000 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.6324 \\ 0.69295 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.7324 \\ 0.7930 \end{bmatrix}_{(2 \times 1)}$$
$$o_2 = \sigma\!\left(\begin{bmatrix} 0.7324 \\ 0.7930 \end{bmatrix}\right) = \begin{bmatrix} \mathbf{0.6753} \\ \mathbf{0.6885} \end{bmatrix}_{(2 \times 1)}$$

### E. Candidate Cell State $\tilde{C}_2 = \tanh(W_c [h_1, x_2] + b_c)$
$$W_c [h_1, x_2] + b_c = \begin{bmatrix} 0.5 & 0.0 & 1.0 & -0.5 \\ 0.0 & 0.5 & -0.5 & 1.0 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.3220 \\ -0.1429 \\ 0.5000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} 0.1610 + 0.0000 + 0.5000 - 0.5000 \\ 0.0000 - 0.07145 - 0.2500 + 1.0000 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.1610 \\ 0.6786 \end{bmatrix}_{(2 \times 1)}$$
$$\tilde{C}_2 = \tanh\!\left(\begin{bmatrix} 0.1610 \\ 0.6786 \end{bmatrix}\right) = \begin{bmatrix} \mathbf{0.1596} \\ \mathbf{0.5906} \end{bmatrix}_{(2 \times 1)}$$

### F. Cell State $C_2$ & Hidden State $h_2$ Updates
$$C_2 = f_2 \odot C_1 + i_2 \odot \tilde{C}_2 = \begin{bmatrix} 0.7523 \\ 0.6961 \end{bmatrix}_{(2 \times 1)} \odot \begin{bmatrix} 0.5255 \\ -0.2541 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.6808 \\ 0.6853 \end{bmatrix}_{(2 \times 1)} \odot \begin{bmatrix} 0.1596 \\ 0.5906 \end{bmatrix}_{(2 \times 1)}$$
$$C_2 = \begin{bmatrix} 0.3953 \\ -0.1769 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.1087 \\ 0.4047 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.5040} \\ \mathbf{0.2278} \end{bmatrix}_{(2 \times 1)}$$

$$h_2 = o_2 \odot \tanh(C_2) = \begin{bmatrix} 0.6753 \\ 0.6885 \end{bmatrix}_{(2 \times 1)} \odot \tanh\!\left(\begin{bmatrix} 0.5040 \\ 0.2278 \end{bmatrix}\right) = \begin{bmatrix} 0.6753 \\ 0.6885 \end{bmatrix}_{(2 \times 1)} \odot \begin{bmatrix} 0.4653 \\ 0.2239 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.3142} \\ \mathbf{0.1542} \end{bmatrix}_{(2 \times 1)}$$

---

# 📄 PAGE 6: Step 2C — LSTM Forward Pass at Time Step $t = 3$ (Token `2`: `"deep"`)

### A. Concatenate Previous Hidden State $h_2$ and Current Embedding $x_3$
$$[h_2, x_3] = \begin{bmatrix} 0.3142 \\ 0.1542 \\ 0.0000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)}$$

### B. Forget Gate $f_3 = \sigma(W_f [h_2, x_3] + b_f)$
$$W_f [h_2, x_3] + b_f = \begin{bmatrix} 0.5 & 0.0 & 0.5 & 0.2 \\ 0.0 & 0.5 & -0.2 & 0.5 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.3142 \\ 0.1542 \\ 0.0000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} (0.5)(0.3142) + (0.0)(0.1542) + (0.5)(0.0) + (0.2)(1.0) \\ (0.0)(0.3142) + (0.5)(0.1542) + (-0.2)(0.0) + (0.5)(1.0) \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} 0.1571 + 0.2000 \\ 0.0771 + 0.5000 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.3571 \\ 0.5771 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.5 \\ 0.5 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.8571 \\ 1.0771 \end{bmatrix}_{(2 \times 1)}$$
$$f_3 = \sigma\!\left(\begin{bmatrix} 0.8571 \\ 1.0771 \end{bmatrix}\right) = \begin{bmatrix} \mathbf{0.7021} \\ \mathbf{0.7459} \end{bmatrix}_{(2 \times 1)}$$

### C. Input Gate $i_3 = \sigma(W_i [h_2, x_3] + b_i)$
$$W_i [h_2, x_3] + b_i = \begin{bmatrix} 0.4 & -0.2 & 0.8 & 0.2 \\ -0.2 & 0.4 & 0.2 & 0.8 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.3142 \\ 0.1542 \\ 0.0000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} (0.4)(0.3142) + (-0.2)(0.1542) + (0.8)(0.0) + (0.2)(1.0) \\ (-0.2)(0.3142) + (0.4)(0.1542) + (0.2)(0.0) + (0.8)(1.0) \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} 0.12568 - 0.03084 + 0.0000 + 0.2000 \\ -0.06284 + 0.06168 + 0.0000 + 0.8000 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.2948 \\ 0.7988 \end{bmatrix}_{(2 \times 1)}$$
$$i_3 = \sigma\!\left(\begin{bmatrix} 0.2948 \\ 0.7988 \end{bmatrix}\right) = \begin{bmatrix} \mathbf{0.5732} \\ \mathbf{0.6897} \end{bmatrix}_{(2 \times 1)}$$

### D. Output Gate $o_3 = \sigma(W_o [h_2, x_3] + b_o)$
$$W_o [h_2, x_3] + b_o = \begin{bmatrix} 0.5 & 0.2 & 0.6 & 0.2 \\ 0.2 & 0.5 & 0.2 & 0.6 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.3142 \\ 0.1542 \\ 0.0000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} 0.1571 + 0.03084 + 0.0000 + 0.2000 \\ 0.06284 + 0.0771 + 0.0000 + 0.6000 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.3879 \\ 0.7399 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.1 \\ 0.1 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} 0.4879 \\ 0.8399 \end{bmatrix}_{(2 \times 1)}$$
$$o_3 = \sigma\!\left(\begin{bmatrix} 0.4879 \\ 0.8399 \end{bmatrix}\right) = \begin{bmatrix} \mathbf{0.6196} \\ \mathbf{0.6984} \end{bmatrix}_{(2 \times 1)}$$

### E. Candidate Cell State $\tilde{C}_3 = \tanh(W_c [h_2, x_3] + b_c)$
$$W_c [h_2, x_3] + b_c = \begin{bmatrix} 0.5 & 0.0 & 1.0 & -0.5 \\ 0.0 & 0.5 & -0.5 & 1.0 \end{bmatrix}_{(2 \times 4)} \begin{bmatrix} 0.3142 \\ 0.1542 \\ 0.0000 \\ 1.0000 \end{bmatrix}_{(4 \times 1)} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}_{(2 \times 1)}$$
$$= \begin{bmatrix} 0.1571 + 0.0000 + 0.0000 - 0.5000 \\ 0.0000 + 0.0771 - 0.0000 + 1.0000 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} -0.3429 \\ 1.0771 \end{bmatrix}_{(2 \times 1)}$$
$$\tilde{C}_3 = \tanh\!\left(\begin{bmatrix} -0.3429 \\ 1.0771 \end{bmatrix}\right) = \begin{bmatrix} \mathbf{-0.3301} \\ \mathbf{0.7921} \end{bmatrix}_{(2 \times 1)}$$

### F. Final Context Cell State $C_3$ & Final Hidden State $h_3$ ($T = 3$)
$$C_3 = f_3 \odot C_2 + i_3 \odot \tilde{C}_3 = \begin{bmatrix} 0.7021 \\ 0.7459 \end{bmatrix}_{(2 \times 1)} \odot \begin{bmatrix} 0.5040 \\ 0.2278 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} 0.5732 \\ 0.6897 \end{bmatrix}_{(2 \times 1)} \odot \begin{bmatrix} -0.3301 \\ 0.7921 \end{bmatrix}_{(2 \times 1)}$$
$$C_3 = \begin{bmatrix} 0.3539 \\ 0.1699 \end{bmatrix}_{(2 \times 1)} + \begin{bmatrix} -0.1892 \\ 0.5463 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.1647} \\ \mathbf{0.7162} \end{bmatrix}_{(2 \times 1)}$$

$$h_3 = o_3 \odot \tanh(C_3) = \begin{bmatrix} 0.6196 \\ 0.6984 \end{bmatrix}_{(2 \times 1)} \odot \tanh\!\left(\begin{bmatrix} 0.1647 \\ 0.7162 \end{bmatrix}\right) = \begin{bmatrix} 0.6196 \\ 0.6984 \end{bmatrix}_{(2 \times 1)} \odot \begin{bmatrix} 0.1632 \\ 0.6145 \end{bmatrix}_{(2 \times 1)} = \begin{bmatrix} \mathbf{0.1011} \\ \mathbf{0.4292} \end{bmatrix}_{(2 \times 1)}$$

---

# 📄 PAGE 7: Step 3 (Logits Layer) & Step 4 (Softmax & Next-Word Prediction)

## 4.1 Step 3: Dense Logits Layer ($y = W_{\text{out}} h_3 + b_{\text{out}}$)

Using the final hidden state $h_T = h_3 = \begin{bmatrix} 0.1011 \\ 0.4292 \end{bmatrix}_{(2 \times 1)}$, we project from hidden size $h = 2$ to vocabulary size $V = 5$:

$$y = W_{\text{out}} h_3 + b_{\text{out}} =
\begin{bmatrix}
0.5 & -1.0 \\
0.8 & 0.2 \\
-0.4 & 0.6 \\
-0.5 & 2.0 \\
-0.2 & -0.8
\end{bmatrix}_{(5 \times 2)}
\begin{bmatrix}
0.1011 \\
0.4292
\end{bmatrix}_{(2 \times 1)} +
\begin{bmatrix}
0.0 \\
0.0 \\
0.1 \\
0.5 \\
-0.2
\end{bmatrix}_{(5 \times 1)}$$

Explicit row-by-row calculation:
- **Row 0 (`"I"`)**: $(0.5)(0.1011) + (-1.0)(0.4292) + 0.0 = 0.05055 - 0.4292 + 0.0 = \mathbf{-0.3787}$
- **Row 1 (`"love"`)**: $(0.8)(0.1011) + (0.2)(0.4292) + 0.0 = 0.08088 + 0.08584 + 0.0 = \mathbf{0.1667}$
- **Row 2 (`"deep"`)**: $(-0.4)(0.1011) + (0.6)(0.4292) + 0.1 = -0.04044 + 0.25752 + 0.1 = \mathbf{0.3171}$
- **Row 3 (`"learning"`)**: $(-0.5)(0.1011) + (2.0)(0.4292) + 0.5 = -0.05055 + 0.8584 + 0.5 = \mathbf{1.3079}$
- **Row 4 (`"<EOS>"`)**: $(-0.2)(0.1011) + (-0.8)(0.4292) - 0.2 = -0.02022 - 0.34336 - 0.2 = \mathbf{-0.5636}$

$$y = \begin{bmatrix}
-0.3787 \\
0.1667 \\
0.3171 \\
\mathbf{1.3079} \\
-0.5636
\end{bmatrix}_{(5 \times 1)}$$

---

## 4.2 Step 4: Softmax Probability Distribution & Token Selection

Applying $\text{Softmax}(y)_k = \frac{e^{y_k}}{\sum_{j=0}^{4} e^{y_j}}$:

1. **Exponentiate Each Logit**:
   - $e^{y_0} = e^{-0.3787} = \mathbf{0.6848}$
   - $e^{y_1} = e^{0.1667} = \mathbf{1.1814}$
   - $e^{y_2} = e^{0.3171} = \mathbf{1.3731}$
   - $e^{y_3} = e^{1.3079} = \mathbf{3.6984}$
   - $e^{y_4} = e^{-0.5636} = \mathbf{0.5692}$

2. **Sum of Exponentials (Partition Function $Z$)**:
   $$\sum_{j=0}^{4} e^{y_j} = 0.6848 + 1.1814 + 1.3731 + 3.6984 + 0.5692 = \mathbf{7.5069}$$

3. **Normalize into Final Probabilities $P \in \mathbb{R}^{5 \times 1}$**:

$$P = \text{Softmax}(y) = \begin{bmatrix}
0.6848 / 7.5069 \\
1.1814 / 7.5069 \\
1.3731 / 7.5069 \\
3.6984 / 7.5069 \\
0.5692 / 7.5069
\end{bmatrix}_{(5 \times 1)} = \begin{bmatrix}
0.0912 \\
0.1574 \\
0.1829 \\
\mathbf{0.4927} \\
0.0758
\end{bmatrix}_{(5 \times 1)}
\begin{array}{l}
\rightarrow \text{ID } 0 \text{ ("I"): } 9.12\% \\
\rightarrow \text{ID } 1 \text{ ("love"): } 15.74\% \\
\rightarrow \text{ID } 2 \text{ ("deep"): } 18.29\% \\
\rightarrow \mathbf{\text{ID } 3 \text{ ("learning"): } 49.27\% \;\; \checkmark \text{ WINNER}} \\
\rightarrow \text{ID } 4 \text{ ("<EOS>"): } 7.58\%
\end{array}$$

*(Sanity Check: $0.0912 + 0.1574 + 0.1829 + 0.4927 + 0.0758 = \mathbf{1.0000}$)*

4. **Final Predicted Next Token**:
   $$\hat{k} = \arg\max_{k \in \{0,1,2,3,4\}} P_k = \mathbf{3} \implies \text{Predicted Next Word} = \mathbf{\text{"learning"}}$$
   $$\text{Completed Sequence: } \mathbf{\text{"I love deep learning"}}$$

---

## 4.3 Summary State Evolution Table Across Context ($t = 1 \to 3$)

| Step $t$ | Token | $x_t$ `(2×1)` | $f_t$ `(2×1)` | $i_t$ `(2×1)` | $\tilde{C}_t$ `(2×1)` | $C_t$ `(2×1)` | $o_t$ `(2×1)` | $h_t$ `(2×1)` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$t=0$** | *(Init)* | — | — | — | — | `[0.0000, 0.0000]` | — | `[0.0000, 0.0000]` |
| **$t=1$** | `"I"` (`0`) | `[1.0, 0.0]` | `[0.7311, 0.5744]` | `[0.6900, 0.5498]` | `[0.7616, -0.4621]` | `[0.5255, -0.2541]` | `[0.6682, 0.5744]` | `[0.3220, -0.1429]` |
| **$t=2$** | `"love"` (`1`) | `[0.5, 1.0]` | `[0.7523, 0.6961]` | `[0.6808, 0.6853]` | `[0.1596, 0.5906]` | `[0.5040, 0.2278]` | `[0.6753, 0.6885]` | `[0.3142, 0.1542]` |
| **$t=3$** | `"deep"` (`2`) | `[0.0, 1.0]` | `[0.7021, 0.7459]` | `[0.5732, 0.6897]` | `[-0.3301, 0.7921]` | `[0.1647, 0.7162]` | `[0.6196, 0.6984]` | `[0.1011, 0.4292]` |
