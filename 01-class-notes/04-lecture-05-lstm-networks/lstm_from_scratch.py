"""
Lecture 05: Long Short-Term Memory (LSTM) from Scratch in Pure Python.
Demonstrates:
1. Exact mathematical LSTM Cell with all 3 Gates (Forget, Input, Output) + Candidate State.
2. The Dual-Memory Streams: Long-Term Cell State (C_t) & Short-Term Working Memory (h_t).
3. Sigmoid gating valves vs. tanh memory additions.
4. Tangible feature tracing across a sample sentence.
"""

import math
import random


# =====================================================================
# ACTIVATION FUNCTIONS & HELPERS
# =====================================================================
def sigmoid(val: float) -> float:
    """Sigmoid valve function: squashes to (0.0, 1.0)."""
    # Clip for numeric stability
    val = max(-500.0, min(500.0, val))
    return 1.0 / (1.0 + math.exp(-val))


def apply_sigmoid(vec: list[float]) -> list[float]:
    return [sigmoid(v) for v in vec]


def apply_tanh(vec: list[float]) -> list[float]:
    return [math.tanh(v) for v in vec]


def vec_add(a: list[float], b: list[float]) -> list[float]:
    """Element-wise vector addition: a + b"""
    return [a[i] + b[i] for i in range(len(a))]


def hadamard_prod(a: list[float], b: list[float]) -> list[float]:
    """Element-wise (Hadamard) vector multiplication: a * b"""
    return [a[i] * b[i] for i in range(len(a))]


def mat_vec_mul(W: list[list[float]], x: list[float]) -> list[float]:
    """Matrix-vector multiplication: W (m x n) @ x (n,) -> (m,)"""
    return [sum(W[i][j] * x[j] for j in range(len(x))) for i in range(len(W))]


# =====================================================================
# LSTM CELL & SEQUENCE MODEL
# =====================================================================
class PurePythonLSTMCell:
    def __init__(self, input_dim: int, hidden_dim: int, seed: int = 42):
        random.seed(seed)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        scale = 1.0 / math.sqrt(hidden_dim)

        def init_matrix(rows, cols):
            return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

        def init_bias(size, default_val=0.0):
            return [default_val for _ in range(size)]

        # 1. Forget Gate Parameters (f_t = sigmoid(W_f @ x + U_f @ h + b_f))
        # Note: Forget gate bias often initialized to +1.0 in practice to prevent early forgetting
        self.W_f = init_matrix(hidden_dim, input_dim)
        self.U_f = init_matrix(hidden_dim, hidden_dim)
        self.b_f = init_bias(hidden_dim, default_val=1.0)

        # 2. Input Gate Parameters (i_t = sigmoid(W_i @ x + U_i @ h + b_i))
        self.W_i = init_matrix(hidden_dim, input_dim)
        self.U_i = init_matrix(hidden_dim, hidden_dim)
        self.b_i = init_bias(hidden_dim, default_val=0.0)

        # 3. Candidate Cell State Parameters (C_tilde = tanh(W_c @ x + U_c @ h + b_c))
        self.W_c = init_matrix(hidden_dim, input_dim)
        self.U_c = init_matrix(hidden_dim, hidden_dim)
        self.b_c = init_bias(hidden_dim, default_val=0.0)

        # 4. Output Gate Parameters (o_t = sigmoid(W_o @ x + U_o @ h + b_o))
        self.W_o = init_matrix(hidden_dim, input_dim)
        self.U_o = init_matrix(hidden_dim, hidden_dim)
        self.b_o = init_bias(hidden_dim, default_val=0.0)

    def step(self, x_t: list[float], h_prev: list[float], C_prev: list[float]):
        """
        Executes one time step of the LSTM Cell:
          1. f_t = sigmoid(W_f @ x_t + U_f @ h_prev + b_f)   [Forget Gate Valve]
          2. i_t = sigmoid(W_i @ x_t + U_i @ h_prev + b_i)   [Input Gate Valve]
          3. C_tilde = tanh(W_c @ x_t + U_c @ h_prev + b_c)  [New Candidate Memory]
          4. C_t = (f_t * C_prev) + (i_t * C_tilde)          [Additive Cell State Highway]
          5. o_t = sigmoid(W_o @ x_t + U_o @ h_prev + b_o)   [Output Gate Valve]
          6. h_t = o_t * tanh(C_t)                           [Short-Term Working State]
        """
        # 1. Forget Gate
        linear_f = vec_add(vec_add(mat_vec_mul(self.W_f, x_t), mat_vec_mul(self.U_f, h_prev)), self.b_f)
        f_t = apply_sigmoid(linear_f)

        # 2. Input Gate
        linear_i = vec_add(vec_add(mat_vec_mul(self.W_i, x_t), mat_vec_mul(self.U_i, h_prev)), self.b_i)
        i_t = apply_sigmoid(linear_i)

        # 3. Candidate State
        linear_c = vec_add(vec_add(mat_vec_mul(self.W_c, x_t), mat_vec_mul(self.U_c, h_prev)), self.b_c)
        c_tilde = apply_tanh(linear_c)

        # 4. Update Long-Term Cell State (Additive Highway)
        retained_past = hadamard_prod(f_t, C_prev)
        new_information = hadamard_prod(i_t, c_tilde)
        C_t = vec_add(retained_past, new_information)

        # 5. Output Gate
        linear_o = vec_add(vec_add(mat_vec_mul(self.W_o, x_t), mat_vec_mul(self.U_o, h_prev)), self.b_o)
        o_t = apply_sigmoid(linear_o)

        # 6. Hidden State (Short-term Output)
        h_t = hadamard_prod(o_t, apply_tanh(C_t))

        return {
            "f_t": f_t,
            "i_t": i_t,
            "c_tilde": c_tilde,
            "C_t": C_t,
            "o_t": o_t,
            "h_t": h_t
        }


# =====================================================================
# DEMO WALKTHROUGH & FEATURE TRACING
# =====================================================================
if __name__ == "__main__":
    print("=" * 80)
    print(" LECTURE 05: LSTM NETWORK STEP-BY-STEP FORWARD PASS & GATE TRACE")
    print("=" * 80)

    # 4-Dimensional semantic feature layout:
    # Dim 0: Gender (+1.0 Male, -1.0 Female)
    # Dim 1: Royalty (+1.0 King/Queen, 0.0 Commoner)
    # Dim 2: Tense (-1.0 Past, 0.0 Present, +1.0 Future)
    # Dim 3: Health Status (+1.0 Healthy, -1.0 Sick)
    dim_names = ["Gender(Male=+1/Fem=-1)", "Royalty(+1)", "Tense(Past=-1/Fut=+1)", "Health(Sick=-1)"]

    input_dim = 4
    hidden_dim = 4

    lstm = PurePythonLSTMCell(input_dim=input_dim, hidden_dim=hidden_dim, seed=10)

    # Initial Memory States (Start of Story)
    C_curr = [0.0] * hidden_dim
    h_curr = [0.0] * hidden_dim

    sentence_tokens = ["Alexander", "was", "a", "great", "king", ".", "Later", "in", "Babylon", "he", "fell", "sick"]
    
    # Generate fixed synthetic embeddings for our demo tokens
    random.seed(42)
    embeddings = {
        token: [round(random.uniform(-1.0, 1.0), 2) for _ in range(input_dim)]
        for token in sentence_tokens
    }
    # Manually craft "Alexander" and "king" to have strong male royalty features
    embeddings["Alexander"] = [0.95, 0.90, 0.0, 0.5]
    embeddings["king"] = [0.90, 0.99, 0.0, 0.2]
    embeddings["sick"] = [0.0, 0.0, -0.8, -0.95]

    print(f"\nProcessing sentence: {' '.join(sentence_tokens)}\n")

    for step, token in enumerate(sentence_tokens):
        x_t = embeddings[token]
        res = lstm.step(x_t, h_curr, C_curr)
        
        # Update current state for next step
        C_curr = res["C_t"]
        h_curr = res["h_t"]

        # Highlight key moments
        if token in ["Alexander", "king", "Babylon", "sick"]:
            print("-" * 80)
            print(f"Step {step+1:2d} | Token: '{token:9s}'")
            print(f"  Input Gate (i_t)   : {[round(v, 2) for v in res['i_t']]}  (How much new fact to write)")
            print(f"  Forget Gate (f_t)  : {[round(v, 2) for v in res['f_t']]}  (How much past diary to retain)")
            print(f"  Cell State (C_t)   : {[round(v, 3) for v in res['C_t']]}  <── [LONG-TERM MEMORY HIGHWAY]")
            print(f"  Hidden State (h_t) : {[round(v, 3) for v in res['h_t']]}  <── [SHORT-TERM WORKING STATE]")

    print("\n" + "=" * 80)
    print(" KEY ARCHITECTURAL TAKEAWAYS:")
    print("=" * 80)
    print("1. THE DUAL HIGHWAY: C_t preserves long-range information across the entire story,")
    print("   while h_t acts as the active working memory for current next-token generation.")
    print("2. SIGMOID VALVES: Gates use Sigmoid in [0, 1] to serve as percentage flow controllers.")
    print("3. GRADIENT HIGHWAY: Backpropagation along C_t uses addition (+), meaning gradients")
    print("   flow across 100+ words without exponentially decaying to zero.")
    print("=" * 80)
