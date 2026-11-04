# Teaching Video Script: Handwritten LSTM for Next-Word Generation
### *12–15 Minute Classroom Walkthrough Strictly Synced to Your 11 Scanned Handwritten Pages*

> **Recording & Pacing Guide**:
> - **Screen Setup**: Open your **11-page scanned handwritten PDF** full-screen with your cursor visible; place your webcam bubble in the bottom-right corner.
> - **Target Length**: **13–14 minutes** (~1,550 spoken words at ~115–120 wpm while pointing to matrices on screen).
> - **Optional 2-Speaker Split (Damodhar & Vamshi)**:
>   - **Speaker 1**: **Pages 1–6** (`00:00 – 07:00`)
>   - **Speaker 2**: **Pages 7–11** (`07:00 – 13:45`)
> - **3 Quick Pointer Tips While Recording**:
>   1. *Page 1 (inside RNN box)*: Read aloud naturally as `tanh(W_h * h_{t-1} + W_x * x_t + b)`.
>   2. *Page 8 (bottom `h_2` line)*: Inside `tanh(...)` `[0.67, 0.68]` was written instead of `C_2 = [0.50, 0.22]`, though the final answer `[0.31, 0.15]` was computed from `tanh([0.50, 0.22])`. Simply point to `C_2 = [0.50, 0.22]` and `h_2 = [0.31, 0.15]` as written in the script below.
>   3. *Page 9 to Page 10 (`f_3`)*: Page 9 ends with pre-activation `[0.8571, 1.0771]` and Page 10 plugs in its Sigmoid output `[0.7021, 0.7459]` — the script bridges this smoothly in one line.

---

## 📄 PAGE 1: From Vanilla RNN Bottlenecks to LSTM Solutions (`00:00 – 01:40`)

**[Cue: Display Page 1 on screen.]**

"Hello everyone! On behalf of Damodhar and Vamshi, welcome to our classroom walkthrough on **Long Short-Term Memory networks—LSTMs—for Next-Word Generation**. Today we will walk directly through our 11 handwritten pages—starting with why Vanilla RNNs struggle, examining the internals of an LSTM cell, and manually computing every matrix multiplication, dimension, gate activation, logit, and Softmax probability across three time steps.

**[Cue: Point to the `RNN CELL` diagram on Page 1.]**

Starting here on **Page 1**, look at the **Vanilla RNN Cell**. At time step `t`, it takes the previous **short memory `h_{t-1}`** from the left and the **current token `x_t`** from the bottom, computing `tanh(W_h * h_{t-1} + W_x * x_t + b)` to output a single vector **`h_t`**—which goes both upward to the logits layer and rightward to the next step.

**[Cue: Point to `BOTTLENECKS` ① and ②.]**

Why does this fail on longer sequences? Look at the two **Bottlenecks** below:
1. **First, Memory Pressure**: `h_t` is a **single overwritten vector**. It must hold both immediate short-term syntax and long-term context in one small buffer. Every new token overwrites the entire state, erasing early words.
2. **Second, the Vanishing Gradient Problem**: During Backpropagation Through Time, gradients repeatedly multiply by `W_h` and the derivative of `tanh`, which is at most `1`. Across many steps, gradients shrink exponentially toward zero, so early tokens stop learning.

**[Cue: Point to `HOW LSTMs Solve This Bottlenecks` ① and ②.]**

LSTMs fix both problems using two ideas:
1. **Dual Stream Memory**—separating long-term **Cell State `C_t`** from short-term **Hidden State `h_t`**, and
2. **Learnable Sigmoid Gates**—valves between `0` and `1` that control what to forget, write, and output."

---

## 📄 PAGE 2: LSTM Cell Internals (`01:40 – 03:20`)

**[Cue: Scroll to Page 2 — Point to the top `High Level View` box.]**

"Moving to **Page 2: LSTM Cell Internals**.
First, look at the **High Level View** at the top:
- Straight across the top runs the **Cell State**—our long-term memory highway. Notice there are no weight matrices directly multiplying this line, only element-wise multiply `(×)` and element-wise add `(+)`. Because of that linear additive path, gradients flow backward through time without vanishing.
- Along the bottom runs our **Hidden State (Short Memory)**, controlled by three gate stages labeled **①**, **②**, and **③**.

**[Cue: Scroll to the lower diagram `Long Term Cell State` on Page 2.]**

Now look at the detailed **dimension diagram** at the bottom of Page 2 for our exact sizes (`h = 2`, `e = 2`):
- At the bottom left, previous hidden state **`h_{t-1}` `(2x1)`** and current token **`x_t` `(2x1)`** stack into **`Concat(h_{t-1}, x_t)` of shape `(4x1)`**.
- Left branch: Sigmoid with `W_f` produces the **Forget Gate `f_t`**, multiplying `C_{t-1}` `(2x1)` at `(×)`.
- Middle branches: Sigmoid with `W_i` gives the **Input Gate**, and `tanh` with `W_c` creates the **Candidate values**. They multiply at `(×)` and add at `(+)` onto the top highway to produce **`C_t` `(2x1)`**.
- Right branch: `C_t` passes through `tanh` and multiplies with the **Output Gate** (`W_o`) to emit **`h_t` `(2x1)`**."

---

## 📄 PAGE 3: The 6 Governing Equations (`03:20 – 04:30`)

**[Cue: Scroll to Page 3 — Point down equations ① to ⑥.]**

"On **Page 3**, we list the **6 Governing Equations** used in every time step:
1. **① Forget Gate**: `f_t = σ(W_f [h_{t-1}, x_t] + b_f)` — how much of old cell memory `C_{t-1}` to keep, between `0` and `1`.
2. **② Input Gate**: `i_t = σ(W_i [h_{t-1}, x_t] + b_i)` — how much new candidate info to write.
3. **③ Output Gate**: `O_t = σ(W_o [h_{t-1}, x_t] + b_o)` — how much updated cell memory to expose as `h_t`.
4. **④ Candidate Cell State**: `C̄_t = tanh(W_c [h_{t-1}, x_t] + b_c)` — new proposed semantic content. Notice we use `tanh` in `(-1, +1)` here so the cell can either add positive signal or subtract negative signal, whereas gates ① to ③ use Sigmoid in `(0, 1)` because they act as percentage valves.
5. **⑤ Cell State Update**: `C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̄_t`, using element-wise multiplication `⊙`.
6. **⑥ Hidden State Update**: `h_t = O_t ⊙ tanh(C_t)`."

---

## 📄 PAGE 4: Problem Settings, Vocabulary & Step 1 Embeddings (`04:30 – 06:00`)

**[Cue: Scroll to Page 4 — Point to settings, table, and `Step 1`.]**

"On **Page 4**, here are the assignment hyperparameters:
- Vocabulary size `V = 5`, Hidden state size `h = 2`, Token embedding size `e = 2`, Context size `c = 3`, giving Concatenated Input dimension `(h + e) = 4`.

In our **Vocabulary Mapping** table:
- `0` = `"I"` (Subject Pronoun)
- `1` = `"love"` (Action / sentiment verb)
- `2` = `"deep"` (Technical Modifier)
- `3` = `"learning"` (Target Technical noun)
- `4` = `"<EOS>"` (End-of-sentence Token)

Our 3-token context is **`["I", "love", "deep"]`**, or Token IDs **`[0, 1, 2]`**.

In **Step 1** at the bottom, we define our `(5x2)` Embedding Matrix **`E`**, where Column 1 captures *Subject / syntax* and Column 2 captures *Sentiment / Technical Topic*. Multiplying `E^T` `(2x5)` by each `(5x1)` one-hot vector gives our three `(2x1)` embeddings:
- `x_1 = [1.0, 0.0]` for `"I"`,
- `x_2 = [0.5, 1.0]` for `"love"`, and
- `x_3 = [0.0, 1.0]` for `"deep"`."

---

## 📄 PAGE 5: Explicit LSTM & Logits Layer Parameters (`06:00 – 07:00`)

**[Cue: Scroll to Page 5.]**

"On **Page 5**, we write out every parameter matrix with explicit dimensions:
- Initial states at `t = 0`: `h_0 = [0.0, 0.0]` `(2x1)` and `C_0 = [0.0, 0.0]` `(2x1)`.
- Each gate weight matrix—**`W_f`, `W_i`, `W_o`, `W_c`**—has shape `(h x (h+e)) = (2x4)` and bias shape `(2x1)`, because `(2x4)` times `(4x1)` plus `(2x1)` produces a `(2x1)` vector. Notice `b_f = [0.5, 0.5]` is initialized positively so the forget gate starts open and retains early context.
- Finally, item 5 defines our **Dense Logits Layer**: **`W_out` of shape `(5x2)`** and **`b_out` of shape `(5x1)`**, projecting hidden size `2` to vocabulary size `5`."

---

## 📄 PAGE 6 & TOP OF PAGE 7: Time Step `t = 1` — Token `0` (`"I"`) (`07:00 – 08:50`)

**[Cue: Scroll to Page 6 — Trace A through E.]**

"Now let's compute **Time Step `t = 1`** for Token `0` (**`"I"`**) on **Page 6**:
- **A) Concatenate `h_0` and `x_1`**: Stacking `[0.0, 0.0]` and `[1.0, 0.0]` gives **`[0.0, 0.0, 1.0, 0.0]` of shape `(4x1)`**. Because only the 3rd entry is `1.0`, each `(2x4)` matrix multiplication cleanly picks its 3rd column:
- **B) Forget Gate `f_1`**: `[0.5, -0.2] + [0.5, 0.5] = [1.0, 0.3]` `(2x1)`. Applying Sigmoid gives **`f_1 = [0.7311, 0.5744]`**.
- **C) Input Gate `i_1`**: Pre-activation `[0.8, 0.2]` `(2x1)` through Sigmoid gives **`[0.69, 0.54]`** (written in compact 2-decimal form on Pages 6–8).
- **D) Output Gate `O_1`**: `[0.6, 0.2] + [0.1, 0.1] = [0.7, 0.3]`, giving Sigmoid **`O_1 = [0.66, 0.57]`** `(2x1)`.
- **E) Candidate `C̄_1`**: `tanh([1.0, -0.5])` gives **`[0.76, -0.46]`** `(2x1)`.

**[Cue: Scroll to top of Page 7 — Section F.]**

At the top of **Page 7 (Section F)**:
- Since `C_0 = [0, 0]`, the first term vanishes and **`C_1 = i_1 ⊙ C̄_1 = [0.52, -0.25]` `(2x1)`**—storing a positive Subject signal `+0.52` in Coordinate 1.
- Then **`h_1 = O_1 ⊙ tanh(C_1) = [0.32, -0.14]` `(2x1)`**."

---

## 📄 BOTTOM OF PAGE 7 & PAGE 8: Time Step `t = 2` — Token `1` (`"love"`) (`08:50 – 10:35`)

**[Cue: Point to bottom half of Page 7 — Time Step `t = 2`, A, B, C.]**

"On the lower half of **Page 7**, we run **Time Step `t = 2`** for Token `1` (**`"love"`**):
- **A) Concatenation**: Stacking `h_1` and `x_2` gives **`[0.32, -0.14, 0.50, 1.00]` `(4x1)`**.
- **B) Forget Gate `f_2`**: `(2x4)` times `(4x1)` plus `[0.5, 0.5]` passed through Sigmoid gives **`f_2 = [0.75, 0.69]` `(2x1)`**—retaining ~75% of our subject memory from step 1!
- **C) Input Gate `i_2`**: Gives **`i_2 = [0.68, 0.68]` `(2x1)`**.

**[Cue: Scroll to Page 8 — D, E, F.]**

Continuing on **Page 8**:
- **D) Output Gate `O_2`**: `σ([0.73, 0.79]) = [0.67, 0.68]` `(2x1)`.
- **E) Candidate `C̄_2`**: `tanh([0.16, 0.67]) = [0.15, 0.59]` `(2x1)`.
- **F) Cell & Hidden Updates**:
  `C_2 = [0.75, 0.69] ⊙ [0.52, -0.25] + [0.68, 0.68] ⊙ [0.15, 0.59] = [0.50, 0.22]` `(2x1)`. Notice Coordinate 1 stays strong at `0.50` while Coordinate 2 flips positive to `+0.22`!
  Taking `tanh(C_2)`—that is `tanh([0.50, 0.22])`—and multiplying element-wise by `O_2 = [0.67, 0.68]` gives **`h_2 = [0.31, 0.15]` `(2x1)`** (or `[0.3142, 0.1542]` to 4 decimals)."

---

## 📄 PAGE 9 & PAGE 10: Time Step `t = 3` — Token `2` (`"deep"`) (`10:35 – 12:20`)

**[Cue: Scroll to Page 9.]**

"Now turn to **Page 9** for our third context word, **Time Step `t = 3`: Token `2` (`"deep"`)**, written to 4 decimal places:
- **Concatenation**: Stacking `h_2 = [0.3142, 0.1542]` and `x_3 = [0.0000, 1.0000]` forms our `(4x1)` input vector.
- **Forget Gate `f_3`**: `(2x4)` times `(4x1)` gives `[0.3571, 0.5771]`, plus `[0.5, 0.5]` equals `[0.8571, 1.0771]` `(2x1)`. Applying Sigmoid gives **`f_3 = [0.7021, 0.7459]`**.
- **Input Gate `i_3`**: Pre-activation `[0.2948, 0.7988]` through Sigmoid gives **`[0.5732, 0.6897]` `(2x1)`**.

**[Cue: Scroll to Page 10.]**

Finishing the gates at the top of **Page 10**:
- **Output Gate `O_3`**: `[0.3879, 0.7399] + [0.1, 0.1] = [0.4879, 0.8399]`, giving Sigmoid **`O_3 = [0.6196, 0.6984]` `(2x1)`**.
- **Candidate `C̄_3`**: `tanh([-0.3429, 1.0771]) = [-0.3301, 0.7921]` `(2x1)`.
- **Final Cell State `C_3` & Final Hidden State `h_3`**:
  `C_3 = [0.7021, 0.7459] ⊙ [0.5040, 0.2279] + [0.5732, 0.6897] ⊙ [-0.3301, 0.7921] = [0.1647, 0.7162]` `(2x1)`.
  Look how Coordinate 2 (*Technical Topic*) surged to **`0.7162`** upon seeing `"deep"`!
  Multiplying `O_3 ⊙ tanh(C_3)` yields our **final context hidden state `h_3 = [0.1011, 0.4292]` `(2x1)`**."

---

## 📄 PAGE 11: Dense Logits Layer, Softmax & Final Prediction (`12:20 – 13:45`)

**[Cue: Scroll to Page 11 — Point to `DENSE LOGITS LAYER` and `SOFTMAX`.]**

"Finally, on **Page 11**, we pass `h_3 = [0.1011, 0.4292]` `(2x1)` into the **Dense Logits Layer**:
1. **`y = W_out * h_3 + b_out`**:
   Multiplying `(5x2)` by `(2x1)` and adding `(5x1)` produces our 5 raw logits:
   `y = [-0.3787, 0.1667, 0.3171, 1.3079, -0.5636]` `(5x1)`.
   Notice Row 4 (Token ID `3`, `"learning"`) has weights `[-0.5, 2.0]` plus bias `0.5`, strongly rewarding `h_3`'s technical component `0.4292` to score the highest logit **`1.3079`**!

2. **Softmax Probability Distribution & Token Selection**:
   Exponentiating each logit and dividing by their sum (`7.5069`) gives `Softmax(y)` `(5x1)`:
   - ID `0` (`"I"`): `0.0912` (`9.12%`)
   - ID `1` (`"love"`): `0.1574` (`15.74%`)
   - ID `2` (`"deep"`): `0.1829` (`18.29%`)
   - **ID `3` (`"learning"`): `0.4927` (`49.27%`) — our clear WINNER!**
   - ID `4` (`"<EOS>"`): `0.0758` (`7.58%`)

Choosing the highest probability gives **Predicted Next Word = `"learning"`**, completing **`"I love deep learning"`**!

Thank you so much for watching our walkthrough!"
