# Teaching Video Script: Handwritten LSTM for Next-Word Generation
### *Professor-to-Undergraduate Classroom Walkthrough (Synced Page-by-Page with Your Scanned PDF)*

> **Recording Setup Tips (Loom / Veed.io / OBS)**:
> - Open your **scanned 7-page handwritten PDF** full-screen on your monitor (with a PDF highlighter/laser pointer cursor active) and keep your **face-cam bubble in the bottom-right corner**.
> - **Target Duration**: ~12–16 minutes (the assignment explicitly notes *"You can make the video as long as possible"* and grades **50% on clarity and depth of video explanation**).
> - Speak warmly and conversationally—as if teaching a sharp 2nd-year Computer Science undergraduate sitting across the desk.

---

## 🎬 Opening & Hook (00:00 – 01:00)

**[Cue: Show Title / Page 1 Header on screen. Smile at the camera.]**

"Hello everyone, welcome to today's session! Today we are going to open up the black box of one of the most important sequence architectures in Deep Learning: the **Long Short-Term Memory network—or LSTM**—and see, down to every single matrix multiplication and decimal place, how it predicts the **next word** in a sentence.

Rather than relying on PyTorch or TensorFlow to hide the linear algebra behind `model.forward()`, we are going to trace an end-to-end Next-Word Generator completely **by hand**:
- With Vocabulary size $V = 5$,
- Token Embedding size $e = 2$,
- Hidden & Cell State size $h = 2$, and
- Context window $c = 3$ words: **`'I'`, `'love'`, `'deep'`**, watching how the gates mathematically evolve step-by-step to predict the fourth word: **`'learning'`**!

Before we compute a single number, let's start on **Page 1** by asking *why* we even need an LSTM in the first place—and what was broken in a standard Vanilla RNN."

---

## 📖 Part 1: Vanilla RNN, Its Bottlenecks, and How LSTM Fixes Them (01:00 – 03:45)

**[Cue: Scroll to Page 1 — Point to Diagram 1 (Vanilla RNN Cell)]**

"Look at this small diagram of a **Vanilla RNN** here on Page 1.
At every time step $t$, a Vanilla RNN takes two things: the current word vector $x_t$ and the previous hidden memory $h_{t-1}$. It multiplies them by weight matrices, adds a bias, and squashes everything through a $\tanh$ activation:
$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$

On the surface, this recurrent loop looks clever. So **what is missing? Why does Vanilla RNN fail on real sentences?** There are **two fundamental bottlenecks**:

1. **First: The Single Mutable Memory Bottleneck—or what I call 'Goldfish Memory'.**
   Notice there is only **one** memory line $h_t$. That single tiny vector has to simultaneously hold *immediate short-term grammar* (like whether the previous word was an adjective) AND *long-term big-picture context* (like who the subject of the paragraph was 30 words ago). Because $h_{t-1}$ gets multiplied by $W_{hh}$ and squashed through $\tanh$ at *every single word*, every new word overwrites the entire memory buffer. There is no lock or shield to protect important early facts.

2. **Second: The Vanishing Gradient Bottleneck during Backpropagation Through Time.**
   **[Cue: Point to the Chain Rule product equation on Page 1]**
   Look at the gradient equation here: when we train the network and send error signals backward from step $T$ back to step $1$, the Chain Rule multiplies the Jacobian $\frac{\partial h_k}{\partial h_{k-1}} = W_{hh}^T \cdot \text{diag}(1 - \tanh^2(z_k))$ over and over again—$T$ times!
   Because the derivative of $\tanh$ lies strictly between $0$ and $1$, multiplying numbers like $0.5 \times 0.5 \times 0.5$ across 20 or 50 steps shrinks the gradient exponentially toward **zero**. Early tokens get zero learning signal!

**[Cue: Scroll to Section 1.2 at bottom of Page 1]**

So **how does LSTM solve both bottlenecks?**
Imagine how *your own human brain* reads a book. You don't overwrite your entire brain every second! You keep **two separate memory tracks**:
1. A **Long-Term Notebook—the Cell State $C_t$**: where you jot down important persistent facts and carry them untouched across pages. Mathematically, $C_t$ is updated by **linear addition** ($+$), not repeated matrix multiplication! That creates a **Constant Error Carousel**—an additive superhighway where gradients flow backward through time without vanishing whenever the forget gate stays near $1$.
2. A **Short-Term Working Memory—the Hidden State $h_t$**: what you need right this second to speak the very next word.

And to control what enters, stays in, or leaves that Long-Term Notebook, LSTM introduces **three Sigmoid Gate valves**."

---

## 🔧 Part 2: Inside the LSTM Cell & Unrolled Architecture (03:45 – 07:00)

**[Cue: Scroll to Page 2 — Point to the LSTM Cell Diagram]**

"Let's zoom inside a single LSTM neuron cluster on **Page 2**. Follow my cursor along the two horizontal lines:

- **At the very top**, see that straight horizontal conveyor belt from $C_{t-1}$ to $C_t$? That is our **Long-Term Cell State Highway**, of shape $(2 \times 1)$. Notice there are **no weight matrices directly blocking this line**—only an element-wise multiply $(\times)$ and an element-wise plus $(+)$!
- **At the bottom**, the previous hidden state $h_{t-1}$ of shape $(2 \times 1)$ and the current word embedding $x_t$ of shape $(2 \times 1)$ meet and stack vertically into a single concatenated vector $[h_{t-1}, x_t]$ of shape **$(4 \times 1)$**.

Now watch that $(4 \times 1)$ vector feed into **four parallel neural layers**:

1. **First, the Forget Gate $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$**:
   Why do we use **Sigmoid ($\sigma$)** here? Because Sigmoid outputs numbers strictly between **$0$ and $1$**! Think of it like a water valve or dimmer switch:
   - $0.0$ means *'Erase 100% of this old memory channel.'*
   - $1.0$ means *'Keep 100% of this old memory intact.'*
   - $0.73$ means *'Retain 73% of the previous memory.'*

2. **Second & Third, the Input Gate $i_t$ ($\sigma$) and Candidate Memory $\tilde{C}_t$ ($\tanh$)**:
   - First, the $\tanh$ layer proposes **new candidate semantic content** $\tilde{C}_t \in (-1, +1)$. Why $\tanh$ instead of Sigmoid? Because memory updates need to be **zero-centered**—we must be able to both *add positive intensity ($+$)* and *subtract/inhibit ($-$)* features without exploding to infinity!
   - Then the **Input Gate $i_t \in (0, 1)$** acts as a gatekeeper deciding *what fraction* of that candidate thought is actually worth writing onto our long-term notebook.
   - Together, they update our Cell State:
     $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
     Notice the symbol $\odot$—that is the **Hadamard (element-wise) product**: $(2 \times 1) \odot (2 \times 1) = (2 \times 1)$.

3. **Fourth, the Output Gate $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$**:
   Our long-term notebook $C_t$ holds *everything* we know so far, but we don't blurt out everything at once when predicting the very next word! So we squash $C_t$ into $(-1, +1)$ using $\tanh(C_t)$ and multiply element-wise by the **Output Gate $o_t \in (0,1)$**:
   $$h_t = o_t \odot \tanh(C_t)$$

**[Cue: Point to Section 2.2 Unrolled Schematic at bottom of Page 2]**

Finally, look at the unrolled architecture at the bottom of Page 2. Since our context window is $c = 3$:
- Word 1 (`'I'`) produces $h_1, C_1$.
- Word 2 (`'love'`) takes $h_1, C_1$ and produces $h_2, C_2$.
- Word 3 (`'deep'`) takes $h_2, C_2$ and produces our **final context hidden state $h_3$** of shape $(2 \times 1)$.
- Only $h_3$ is passed into our **Dense Logits Layer** ($y = W_{\text{out}} h_3 + b_{\text{out}}$) of shape $(5 \times 1)$, followed by **Softmax** to pick the highest-probability next word out of our 5-word vocabulary!"

---

## 📐 Part 3A: Problem Setup, Dimensions & Embeddings (07:00 – 09:00)

**[Cue: Scroll to Page 3]**

"Let's set up our exact matrices and dimensions on **Page 3**. Always check dimensions first—in deep learning, tracking shapes prevents 95% of bugs!

- Vocabulary size $V = 5$:
  - `0`: `'I'` (Subject pronoun)
  - `1`: `'love'` (Positive action verb)
  - `2`: `'deep'` (Technical modifier)
  - `3`: `'learning'` (Target technical noun)
  - `4`: `'<EOS>'` (End of sentence)
- Context sequence ($c = 3$): **`'I'`, `'love'`, `'deep'`**, which maps to Token IDs **`[0, 1, 2]`**.

**[Cue: Point to Embedding Matrix $E$ $(5 \times 2)$ on Page 3]**

In **Step 1 (Token Embedding Layer)**, instead of sparse 5-dimensional one-hot vectors that treat all words as equally unrelated, we map each token ID to a dense embedding vector of size $e = 2$ using our $(5 \times 2)$ Embedding Matrix $E$:
- Think of Coordinate 1 as the *Subject / Grammatical Role axis*, and Coordinate 2 as the *Sentiment / Technical Concept axis*.
- Looking up rows `0`, `1`, and `2` (or equivalently multiplying $E^T$ of shape $(2 \times 5)$ by each $(5 \times 1)$ one-hot vector) gives our three $(2 \times 1)$ input column vectors:
  $$x_1 = \begin{bmatrix} 1.0 \\ 0.0 \end{bmatrix}, \quad x_2 = \begin{bmatrix} 0.5 \\ 1.0 \end{bmatrix}, \quad x_3 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$$

**[Cue: Point to Weight Matrices on Page 3]**

Now check the shapes of our LSTM weights:
- Since $h_{t-1}$ is $(2 \times 1)$ and $x_t$ is $(2 \times 1)$, stacking them gives $[h_{t-1}, x_t]$ of shape **$(4 \times 1)$**.
- To produce a gate vector of hidden size $h = 2$, every weight matrix—$W_f, W_i, W_o, W_c$—must have shape **$(2 \times 4)$**, because $(2 \times 4) \times (4 \times 1) = (2 \times 1)$!
- Notice we initialized the Forget Gate bias $b_f = [0.5, 0.5]^T$ slightly positive—a classic best practice taught by Jozefowicz et al. so the LSTM defaults to *remembering* rather than erasing at initialization.
- And we start at $t = 0$ with clean zero vectors: $h_0 = [0, 0]^T$ and $C_0 = [0, 0]^T$."

---

## 🧮 Part 3B: Explicit Walkthrough of Time Steps $t = 1, 2, 3$ (09:00 – 13:30)

**[Cue: Scroll to Page 4 — Time Step $t = 1$ (Token `0` = `'I'`)]**

"Let's trace **Time Step $t = 1$** on **Page 4**, where our model reads the first word **`'I'`** ($x_1 = [1.0, 0.0]^T$).

1. **Concatenation**: Stacking $h_0 = [0, 0]^T$ above $x_1 = [1.0, 0.0]^T$ gives $[h_0, x_1] = [0.0, 0.0, 1.0, 0.0]^T$ of shape $(4 \times 1)$. Notice how nice this first step is: only the 3rd entry is $1.0$, so matrix-vector multiplication simply picks the 3rd column of each weight matrix!
2. **Forget Gate $f_1$**:
   - $W_f [h_0, x_1] + b_f = [0.5, -0.2]^T + [0.5, 0.5]^T = [1.0, 0.3]^T$.
   - Applying Sigmoid element-wise: $\sigma(1.0) = \frac{1}{1+e^{-1}} = \mathbf{0.7311}$, and $\sigma(0.3) = \mathbf{0.5744}$.
3. **Input Gate $i_1$**:
   - $W_i [h_0, x_1] + b_i = [0.8, 0.2]^T \xrightarrow{\sigma} i_1 = [\mathbf{0.6900}, \mathbf{0.5498}]^T$.
4. **Output Gate $o_1$**:
   - $W_o [h_0, x_1] + b_o = [0.6, 0.2]^T + [0.1, 0.1]^T = [0.7, 0.3]^T \xrightarrow{\sigma} o_1 = [\mathbf{0.6682}, \mathbf{0.5744}]^T$.
5. **Candidate Cell $\tilde{C}_1$**:
   - $W_c [h_0, x_1] + b_c = [1.0, -0.5]^T \xrightarrow{\tanh} \tilde{C}_1 = [\mathbf{0.7616}, \mathbf{-0.4621}]^T$.
6. **Updating $C_1$ and $h_1$**:
   - Since $C_0 = [0, 0]^T$, the forget term $f_1 \odot C_0$ vanishes, leaving $C_1 = i_1 \odot \tilde{C}_1 = [0.6900 \times 0.7616, \; 0.5498 \times (-0.4621)]^T = [\mathbf{0.5255}, \mathbf{-0.2541}]^T$.
   - Look at what $C_1$ just stored! Coordinate 1 (*Subject presence*) jumped to $+0.5255$, while Coordinate 2 (*Technical noun topic*) is negative ($-0.2541$) because we've only seen the pronoun `'I'`.
   - Finally, $h_1 = o_1 \odot \tanh(C_1) = [0.6682 \times 0.4819, \; 0.5744 \times (-0.2488)]^T = [\mathbf{0.3220}, \mathbf{-0.1429}]^T$.

---

**[Cue: Scroll to Page 5 — Time Step $t = 2$ (Token `1` = `'love'`)]**

Now move to **Page 5: Time Step $t = 2$**, where the model reads **`'love'`** ($x_2 = [0.5, 1.0]^T$).

- **Concatenation**: Stacking our new $h_1$ and $x_2$ gives $[0.3220, -0.1429, 0.5000, 1.0000]^T$ of shape $(4 \times 1)$.
- Let's trace the explicit row-by-column dot products:
  - **Forget Gate $f_2$**: Row 1 gives $(0.5)(0.3220) + 0 + (0.5)(0.5) + (0.2)(1.0) + 0.5 = 1.1110 \xrightarrow{\sigma} \mathbf{0.7523}$. Row 2 gives $0 + (0.5)(-0.1429) + (-0.2)(0.5) + (0.5)(1.0) + 0.5 = 0.8286 \xrightarrow{\sigma} \mathbf{0.6961}$. Notice the Forget Gate stays around **70–75% open**—preserving our memory that `'I'` was the subject!
  - **Input Gate $i_2$**: Pre-activations $[0.7574, 0.7784]^T \xrightarrow{\sigma} [\mathbf{0.6808}, \mathbf{0.6853}]^T$.
  - **Output Gate $o_2$**: Pre-activations $[0.7324, 0.7930]^T \xrightarrow{\sigma} [\mathbf{0.6753}, \mathbf{0.6885}]^T$.
  - **Candidate $\tilde{C}_2$**: Pre-activations $[0.1610, 0.6786]^T \xrightarrow{\tanh} [\mathbf{0.1596}, \mathbf{0.5906}]^T$.
- **Watch the Cell State Update $C_2 = f_2 \odot C_1 + i_2 \odot \tilde{C}_2$**:
  $$C_2 = \begin{bmatrix} 0.3953 \\ -0.1769 \end{bmatrix} + \begin{bmatrix} 0.1087 \\ 0.4047 \end{bmatrix} = \begin{bmatrix} \mathbf{0.5040} \\ \mathbf{0.2278} \end{bmatrix}$$
  Look how Coordinate 1 stayed strong ($+0.5040$), while Coordinate 2 flipped from negative to positive ($+0.2278$)!
- Multiplying $o_2 \odot \tanh(C_2)$ yields $h_2 = [\mathbf{0.3142}, \mathbf{0.1542}]^T$.

---

**[Cue: Scroll to Page 6 — Time Step $t = 3$ (Token `2` = `'deep'`)]**

Now let's trace the final context word on **Page 6: Time Step $t = 3$**, reading **`'deep'`** ($x_3 = [0.0, 1.0]^T$).

- **Concatenation**: $[h_2, x_3] = [0.3142, 0.1542, 0.0000, 1.0000]^T$ of shape $(4 \times 1)$.
- Computing our four $(2 \times 4) \times (4 \times 1)$ transformations:
  - **Forget Gate $f_3$**: $\sigma([0.8571, 1.0771]^T) = [\mathbf{0.7021}, \mathbf{0.7459}]^T$.
  - **Input Gate $i_3$**: $\sigma([0.2948, 0.7988]^T) = [\mathbf{0.5732}, \mathbf{0.6897}]^T$.
  - **Output Gate $o_3$**: $\sigma([0.4879, 0.8399]^T) = [\mathbf{0.6196}, \mathbf{0.6984}]^T$.
  - **Candidate $\tilde{C}_3$**: $\tanh([-0.3429, 1.0771]^T) = [\mathbf{-0.3301}, \mathbf{0.7921}]^T$.
- **Look at our Final Cell State $C_3$**:
  $$C_3 = \begin{bmatrix} 0.3539 \\ 0.1699 \end{bmatrix} + \begin{bmatrix} -0.1892 \\ 0.5463 \end{bmatrix} = \begin{bmatrix} \mathbf{0.1647} \\ \mathbf{0.7162} \end{bmatrix}$$
  Look at the story inside $C_3$: Coordinate 2 (*Technical Topic Intensity*) just surged all the way to **$+0.7162$** because `'deep'` strongly signals a technical AI phrase (`'deep ___'`), while Coordinate 1 (*Subject focus*) gently stepped back to $+0.1647$!
- Passing $C_3$ through $\tanh$ and multiplying by $o_3$ produces our **final context hidden state**:
  $$h_3 = \begin{bmatrix} \mathbf{0.1011} \\ \mathbf{0.4292} \end{bmatrix}_{(2 \times 1)}$$"

---

## 🎯 Part 4: Logits Layer, Softmax & Final Prediction (13:30 – 16:00)

**[Cue: Scroll to Page 7 — Logits, Softmax & Summary Table]**

"We are now on the home stretch on **Page 7**!
We have compressed our 3-word context **`'I love deep'`** into $h_3 = [0.1011, 0.4292]^T$ of shape $(2 \times 1)$. How do we turn a 2-number hidden vector into a choice across $V = 5$ vocabulary words?

1. **Step 3 — The Dense Logits Layer ($y = W_{\text{out}} h_3 + b_{\text{out}}$)**:
   - We multiply $W_{\text{out}}$ of shape **$(5 \times 2)$** by $h_3$ of shape **$(2 \times 1)$** and add $b_{\text{out}}$ of shape **$(5 \times 1)$**, giving a **$(5 \times 1)$ vector of raw unnormalized scores called Logits**:
     - Token `0` (`'I'`): $-0.3787$
     - Token `1` (`'love'`): $+0.1667$
     - Token `2` (`'deep'`): $+0.3171$
     - **Token `3` (`'learning'`)**: $(-0.5)(0.1011) + (2.0)(0.4292) + 0.5 = \mathbf{+1.3079}$
     - Token `4` (`'<EOS>'`): $-0.5636$
   - Why can't we stop at raw logits? Because logits can be negative and don't sum to $1$—we need calibrated probabilities!

2. **Step 4 — Softmax Normalization & Argmax Selection**:
   - Softmax exponentiates every logit ($e^{y_k}$)—making every value strictly positive and amplifying our top score $e^{1.3079} = \mathbf{3.6984}$—and divides by the total sum $\sum e^{y_j} = \mathbf{7.5069}$:
     - `'I'`: $9.12\%$
     - `'love'`: $15.74\%$
     - `'deep'`: $18.29\%$
     - **`'learning'` (ID `3`): $49.27\%$**
     - `'<EOS>'`: $7.58\%$
   - Notice all five probabilities sum to **$1.0000$ ($100\%$)**!
   - Taking $\arg\max$ picks **Token ID `3` (`'learning'`)** with **$49.27\%$ confidence**, completing our sentence: **`'I love deep learning'`**!

**[Cue: Point to Section 4.3 Summary Table at the bottom of Page 7]**

To wrap up: look at the State Evolution Table at the bottom of Page 7. Unlike a Vanilla RNN that blindly overwrites memory and suffers from vanishing gradients, our LSTM used its **Forget, Input, and Output Gates** to smoothly carry context across three steps along the **Cell State Highway $C_t$**, cleanly projecting $h_3$ into vocabulary space to generate the exact next word.

Thank you so much for watching, and I hope this step-by-step walkthrough made every matrix, gate, and dimension of LSTMs crystal clear!"
