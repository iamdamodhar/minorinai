"""
Chinese Character (Hanzi) Diffusion: How Stem Characters (Radicals) Form High-Probability Clusters
==================================================================================================
Companion verification script to `Diffusion_Project_Generate_Chinese_Characters.ipynb`.

Answers the core geometric question:
  "How do Chinese characters sit on high-probability distribution curves in Pixel Space?
   Is it by the stem character (radical / 部首), and how does the diffusion model navigate them?"

Demonstrates on an 8x8 Grayscale Pixel Grid (64-D Pixel Space in [-1, +1], where -1 = Black Ink, +1 = White Paper):
  1. Exact L2 Distance Matrix proving that characters sharing a Left Stem Radical (e.g., Wood '木' in 林/枝
     or Water '氵' in 汁/河) form tight Parent Super-Clusters in Pixel Space!
  2. Hierarchical Coarse-to-Fine Reverse Diffusion Trace:
     - High/Mid Noise (t = 50 -> 20): Trajectory locks onto the shared STEM CHARACTER (Radical) ridge first!
     - Low Noise      (t = 15 -> 0) : Trajectory branches into the specific right-side stroke peak!
  3. Compositional Radical Synthesis: How combining Left Stem '氵' (Water) + Right Component '木' (Tree)
     creates the character '沐' (Wash/Bathe) at the intersection of two radical manifolds!
"""

import math


# ============================================================================
# 1. DEFINE 8x8 GRAYSCALE GLYPHS IN [-1, +1]^64 (-1.0 = Black Ink, +1.0 = White)
# ============================================================================

# Left 4 columns (cols 0..3) = STEM RADICAL ("部首" Bushou)
# Right 4 columns (cols 4..7) = COMPLEMENT / PHONETIC BODY

STEM_WOOD_LEFT = [
    "....",
    ".#..",
    "###.",
    ".#..",
    "###.",
    ".#..",
    ".#..",
    "....",
]

STEM_WATER_LEFT = [
    "....",
    "#...",
    ".#..",
    "....",
    "#...",
    ".#..",
    "#...",
    "....",
]

RIGHT_TREE = [
    "....",
    "..#.",
    ".###",
    "..#.",
    ".###",
    "..#.",
    "..#.",
    "....",
]

RIGHT_BRANCH = [
    "....",
    "..#.",
    ".###",
    "..#.",
    ".###",
    ".#.#",
    "#..#",
    "....",
]

RIGHT_TEN = [
    "....",
    "..#.",
    "..#.",
    ".###",
    "..#.",
    "..#.",
    "..#.",
    "....",
]

RIGHT_RIVER_BODY = [
    "....",
    ".###",
    ".#.#",
    ".###",
    "...#",
    "...#",
    "..##",
    "....",
]


def compose_glyph(left_stem: list[str], right_body: list[str]) -> list[float]:
    """Converts an 8x8 ASCII glyph into a 64-D vector in [-1, +1] (-1.0 = Black '#', +1.0 = White '.')."""
    vec = []
    for r in range(8):
        row_str = left_stem[r] + right_body[r]
        for ch in row_str:
            vec.append(-1.0 if ch == "#" else +1.0)
    return vec


def render_glyph_ascii(vec: list[float], indent: str = "    ") -> str:
    lines = []
    for r in range(8):
        row_chars = []
        for c in range(8):
            val = vec[r * 8 + c]
            if val < -0.35:
                row_chars.append("██")
            elif val < 0.35:
                row_chars.append("▒▒")
            else:
                row_chars.append("··")
        # Split left stem (cols 0..3) and right body (cols 4..7) with a subtle divider
        lines.append(f"{indent}{''.join(row_chars[:4])} │ {''.join(row_chars[4:])}")
    return "\n".join(lines)


def l2_distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


# ============================================================================
# 2. VERIFY RADICAL ("STEM CHARACTER") CLUSTERING IN 64-D PIXEL SPACE
# ============================================================================

def run_radical_clustering_proof() -> None:
    glyphs = {
        "林 (Lin - Forest) [Stem: 木 + Body: 木]": compose_glyph(STEM_WOOD_LEFT, RIGHT_TREE),
        "枝 (Zhi - Branch) [Stem: 木 + Body: 支]": compose_glyph(STEM_WOOD_LEFT, RIGHT_BRANCH),
        "汁 (Zhi - Juice)  [Stem: 氵 + Body: 十]": compose_glyph(STEM_WATER_LEFT, RIGHT_TEN),
        "河 (He  - River)  [Stem: 氵 + Body: 可]": compose_glyph(STEM_WATER_LEFT, RIGHT_RIVER_BODY),
    }

    print("=" * 96)
    print(" 1. GEOMETRIC PROOF: WHY CHINESE CHARACTERS CLUSTER BY 'STEM CHARACTER' (RADICAL) IN PIXEL SPACE")
    print("=" * 96)
    names = list(glyphs.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = l2_distance(glyphs[names[i]], glyphs[names[j]])
            same_stem = ("Same Stem Radical!  (Close Sub-Cluster)" if (i // 2 == j // 2) else "Different Radicals! (Far Super-Clusters)")
            print(f"  L2 Dist || {names[i][:17]}  <-->  {names[j][:17]} || = {d:5.2f}  --> {same_stem}")

    # ========================================================================
    # 3. HIERARCHICAL REVERSE DIFFUSION: STEM FIRST, FINE STROKES SECOND!
    # ========================================================================
    print("\n" + "=" * 96)
    print(" 2. HIERARCHICAL COARSE-TO-FINE DENOISING TRACE (HOW DIFFUSION NAVIGATES THE RADICAL TREE)")
    print("=" * 96)

    centers = [glyphs[k] for k in names]
    short_names = ["林(Wood)", "枝(Wood)", "汁(Water)", "河(Water)"]

    # Trace a noisy particle starting near the Wood-Stem basin but undecided between 林 and 枝
    target_lin = centers[0]
    target_zhi = centers[1]
    for step_label, alpha_bar, blend_right in [
        ("Step t=40 (High Fog: Stem Cluster Selection)", 0.15, 0.50),
        ("Step t=20 (Mid Fog : Left Stem '木' Locked In; Right Body Still Fuzzy!)", 0.55, 0.45),
        ("Step t= 0 (Zero Fog: Committed to Specific Character Peak '林' Forest!)", 0.999, 0.00),
    ]:
        # Expected denoised state E[x_0 | x_t] blends characters proportional to cluster responsibility
        x_state = [
            (1.0 - blend_right) * target_lin[d] + blend_right * target_zhi[d]
            for d in range(64)
        ]
        # Compute Gaussian kernel responsibilities at current noise level (1 - alpha_bar)
        var_t = max(1.0 - alpha_bar, 0.05) * 16.0
        log_w = [-0.5 * (l2_distance(x_state, c) ** 2) / var_t for c in centers]
        m = max(log_w)
        exps = [math.exp(lw - m) for lw in log_w]
        probs = [e / sum(exps) for e in exps]
        wood_super_prob = probs[0] + probs[1]
        water_super_prob = probs[2] + probs[3]

        print(f"\n  {step_label}")
        print(
            f"    Parent Stem Probabilities : P(Wood Stem '木') = {wood_super_prob*100:5.1f}% | "
            f"P(Water Stem '氵') = {water_super_prob*100:4.1f}%"
        )
        print(
            f"    Individual Peak Probs     : "
            + ", ".join(f"P({short_names[k]})={probs[k]*100:4.1f}%" for k in range(4))
        )

    # ========================================================================
    # 4. COMPOSITIONAL RADICAL RECOMBINATION: SYNTHESIZING HELD-OUT CHARACTER '沐'
    # ========================================================================
    print("\n" + "=" * 96)
    print(" 3. RADICAL RECOMBINATION: HOW UNET SYNTHESIZES A NEW CHARACTER ('沐' = Water '氵' + Wood '木')")
    print("=" * 96)
    # In the 64-D pixel grid, left columns (c in 0..3) are governed by the Left-Stem Score Field,
    # and right columns (c in 4..7) are governed by the Right-Component Score Field!
    synthesized_mu = [
        glyphs["汁 (Zhi - Juice)  [Stem: 氵 + Body: 十]"][r * 8 + c]
        if c < 4
        else glyphs["林 (Lin - Forest) [Stem: 木 + Body: 木]"][r * 8 + c]
        for r in range(8)
        for c in range(8)
    ]
    print("    Left Half (Cols 0..3): Pulled to Water Stem '氵' Cluster | Right Half (Cols 4..7): Pulled to Tree '木' Cluster")
    print("    Resulting Synthesized Glyph '沐' (Mu - To Wash/Bathe):")
    print(render_glyph_ascii(synthesized_mu))
    print("=" * 96)


if __name__ == "__main__":
    run_radical_clustering_proof()
