# Lecture 04: Code an RNN from Scratch
### *Tiny Encoder-Decoder Implementation, Training Loops, Gradient Descent, Optimization Landscapes, RNN Failure Modes, and the Evolution to LSTMs*

---

## 🧠 Tier 1: The Childhood & Class X Sensory Intuition
*(The Feynman Foundation: The Human Brain Learning Grammar Through Feedback)*

### The Child Learning a Second Language Through Correction

Imagine a young child learning how to translate *"I go"* into Hindi:

$$\text{Source (English): } \text{"I go <EOS>"} \quad \xrightarrow{\text{Translation}} \quad \text{Target (Hindi): } \text{"मैं जाता हूँ <EOS>"}$$

1. **The First Attempt (Untrained Brain)**:
   The child guesses randomly: *"I go"* $\to$ *"आप खाता"* (You eat).
2. **The Parent's Correction (Teacher Forcing & Loss Calculation)**:
   The parent doesn't let the child spiral into wrong guesses. At each word, the parent provides the correct word:
   - *"When you start, say 'मैं'."* $\to$ Brain computes error distance (Cross-Entropy Loss).
   - *"After 'मैं', say 'जाता'."* $\to$ Brain computes error distance.
   - *"After 'जाता', say 'हूँ'."* $\to$ Brain computes error distance.
3. **The Adjustment (Backpropagation & Gradient Descent)**:
   The child's neural synapses adjust slightly (nudging weights $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$) so that next time, the probability of saying *"मैं जाता हूँ"* increases.

```mermaid
flowchart LR
    subgraph TeacherForcing["Parental Guidance / Teacher Forcing Training Loop"]
        direction TB
        EncIn["Source: 'I go <EOS>'"] --> Encoder["Tiny Encoder"]
        Encoder -->|Context Vector (h)| Decoder["Tiny Decoder"]
        
        DecIn["Ground-Truth Target:<br>'<GO>', 'मैं', 'जाता', 'हूँ'"] --> Decoder
        Decoder --> Preds["Predicted Logits"]
        
        Preds --> Loss["Cross-Entropy Loss<br>(Error Distance)"]
        Loss --> Backprop["Backpropagation Through Time<br>(Synaptic Weight Adjustment)"]
    end
```

---

## ⚙️ Tier 2: The Backend Architect's Rosetta Stone
*(The Distributed Systems Lens: Mapping Training & Optimization to Backend Workflows)*

| AI / Training Concept | Distributed Systems & Backend Engineering Equivalent |
| :--- | :--- |
| **Forward Pass** | A **Synchronous API Execution Graph**: Invoking microservices sequentially to compute a response payload. |
| **Teacher Forcing** | An **Idempotent Mock Replay / Golden Test Harness**: Injecting golden reference data at each step to prevent cascading failure cascades during pipeline calibration. |
| **Cross-Entropy Loss** | A **SLA Deviation / Error Metric**: Measuring the distance between actual service output and target SLA. |
| **Gradient ($\nabla_\theta \mathcal{L}$)** | A **Distributed Telemetry Error Vector**: Tracing how much each service configuration parameter contributed to the overall error. |
| **Gradient Descent Step** | An **Autotuning Control Loop**: Modifying configuration dials ($\theta \leftarrow \theta - \eta \nabla \mathcal{L}$) to drive system latency/error to a minimum. |
| **Local Minima & Saddle Points** | **Suboptimal Resource Allocations & Deadlocks**: Reaching a configuration where local throughput improvements stall despite remaining far from optimal global capacity. |

---

## 📐 Tier 3: Geometric & Tangible Domain Feature Space
*(Demystifying Vectors into Real-World Domain Entity Coordinates)*

### 1. Token Vocabulary Coordinate Mapping
In our tiny model:
- English Vocabulary ($V_{src} = 3$): `{"I": 0, "go": 1, "<EOS>": 2}`
- Hindi Vocabulary ($V_{tgt} = 5$): `{"<GO>": 0, "मैं": 1, "जाता": 2, "हूँ": 3, "<EOS>": 4}`

$$\mathbf{x}_{\text{"I"}} = [0.22]^T \in \mathbb{R}^1, \quad \mathbf{h}_{\text{Context}} = \begin{bmatrix} \text{Subject Person Coordinate: } +0.85 \text{ (1st Person)} \\ \text{Action Tense Coordinate: } +0.45 \text{ (Present Habitual)} \end{bmatrix} \in \mathbb{R}^2$$

---

## 🔬 Tier 4: First-Principles Mathematical Rigor & Derivations

### A. Mathematical Formulation of the Pipeline

1. **TinyEncoder**:
   $$\mathbf{h}_t = \tanh\left(\mathbf{W}_h \mathbf{h}_{t-1} + \mathbf{W}_x \mathbf{x}_t + \mathbf{b}\right) \quad (\mathbf{W}_h \in \mathbb{R}^{2 \times 2}, \mathbf{W}_x \in \mathbb{R}^{2 \times 1}, \mathbf{b} \in \mathbb{R}^2)$$
   - Initial state $\mathbf{h}_{-1} = [0, 0]^T$. Final state $\mathbf{h}_{final} = \mathbf{h}_2 \in \mathbb{R}^2$.
2. **TinyDecoder (with Teacher Forcing)**:
   $$\mathbf{h}_t^{dec} = \tanh\left(\mathbf{W}_h \mathbf{h}_{t-1}^{dec} + \mathbf{W}_x \mathbf{x}_t^{dec} + \mathbf{b}\right) \quad (\mathbf{h}_{-1}^{dec} = \mathbf{h}_{final})$$
   $$\mathbf{z}_t = \mathbf{W}_{out} \mathbf{h}_t^{dec} + \mathbf{b}_{out} \in \mathbb{R}^5 \quad (\mathbf{W}_{out} \in \mathbb{R}^{5 \times 2}, \mathbf{b}_{out} \in \mathbb{R}^5)$$
3. **Cross-Entropy Loss**:
   $$\mathcal{L} = \frac{1}{T} \sum_{t=1}^4 \left( -z_{t,\text{target}} + \log \sum_{j=1}^5 e^{z_{t,j}} \right)$$
4. **Gradient Descent Update**:
   $$\theta \leftarrow \theta - \eta \cdot \frac{\partial \mathcal{L}}{\partial \theta}$$

---

### B. Optimization Landscape: Local Minima, Maxima & Saddle Points

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

---

## 💻 Tier 5: Dual-Language Implementation (Python + Java)

### 1. PyTorch Implementation
Available at: [`01-class-notes/03-lecture-04-code-rnn-from-scratch/tiny_rnn_translator_pytorch.py`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/03-lecture-04-code-rnn-from-scratch/tiny_rnn_translator_pytorch.py)

### 2. Pure Python (Zero-Dependency) Implementation
Available at: [`01-class-notes/03-lecture-04-code-rnn-from-scratch/tiny_rnn_translator_pure_python.py`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/03-lecture-04-code-rnn-from-scratch/tiny_rnn_translator_pure_python.py)

### 3. Enterprise Java 17+ Implementation
Available at: [`01-class-notes/03-lecture-04-code-rnn-from-scratch/TinySeq2SeqTranslator.java`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/03-lecture-04-code-rnn-from-scratch/TinySeq2SeqTranslator.java)

---

## 🏢 Tier 6: Enterprise Retail Production Systems Blueprint & GenAI Frontier

### 1. The Context Bottleneck & The Evolution to LSTMs
- In our code, all semantic nuance of `"I go <EOS>"` was compressed into **2 numbers** ($\mathbf{h} \in \mathbb{R}^2$).
- In long sequences, repeated multiplication by $\mathbf{W}_h$ causes gradients to vanish ($\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} \to 0$), making long-range learning impossible.
- **The LSTM Solution (1997)**: Introduced an additive **Cell State ($C_t$)** that acts as an uninterrupted Write-Ahead Log (WAL), eliminating vanishing gradients and allowing memory to span 100+ tokens.
