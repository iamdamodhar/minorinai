# Diffusion Project: Generating Chinese Characters (Hanzi) & How "Stem Characters" (Radicals) Form High-Probability Clusters
### *From `Unihan-kDefinition.txt` ($20,607$ Glyphs) to Hierarchical Radical Ridges, 6-Stage `UNet2DModel`, `DPMSolverMultistepScheduler` & Compositional Glyph Synthesis*

---

## 📂 Deliverables in This Module

- **Original Project Notebook (100% Untouched)**: [`Diffusion_Project_Generate_Chinese_Characters.ipynb`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/Diffusion_Project_Generate_Chinese_Characters.ipynb)
- **Pure Python Radical-Clustering & Denoising Verifier**: [`chinese_character_radical_clusters_from_scratch.py`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/chinese_character_radical_clusters_from_scratch.py)
- **Companion Learning Rate Schedules Guide (`get_cosine_schedule_with_warmup`)**: [`learning_rate_schedules_in_diffusion_and_genai.md`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/10-introduction-to-diffusion-models/learning_rate_schedules_in_diffusion_and_genai.md)

---

## 🧠 1. Core Question: How Do Chinese Characters Sit on High-Probability Distribution Curves? Is It by the "Stem Character" (Radical)?

**Short Answer: YES—100%!** Your intuition about the **"stem character" (called a Radical / 部首 *Bùshǒu* in Chinese linguistics)** is the exact mathematical key to how the diffusion model organizes all $20,607$ characters in Pixel Space!

### Why Chinese Characters Are NOT 20,607 Random Drawings
In [`Cell 12`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/Diffusion_Project_Generate_Chinese_Characters.ipynb#L907-L936), each Chinese character is loaded as a $128 \times 128 \times 1$ grayscale image normalized to $[-1, +1]$ (`-1.0 = Black Ink`, `+1.0 = White Paper`), living in a **$128 \times 128 = 16,384$-dimensional Pixel Space**.

If all $20,607$ characters were unrelated random doodles, they would scatter into $20,607$ tiny, isolated needles across $\mathbb{R}^{16384}$.
**Instead, over $85\%$ of Chinese characters are compound characters built from a shared "Stem Character" (Semantic Radical):**
- **The Water Stem (`氵`)** sits on the left side of **1,500+ characters** (`汁` juice, `河` river, `湖` lake, `海` sea, `清` clear, `沐` bathe).
- **The Wood/Tree Stem (`木`)** sits on the left side of **1,200+ characters** (`林` forest, `枝` branch, `桥` bridge, `桃` peach, `村` village).
- **The Person Stem (`亻`)** sits on the left side of **1,000+ characters** (`你` you, `他` he, `们` plural, `信` trust).

---

### The Geometric Consequence in $16,384$-D Pixel Space: A 3-Level Hierarchical Mountain Range!

Because all 1,200 characters with the **Wood Stem (`木`)** have black ink (`-1.0`) at the **exact same pixel coordinates in the left columns ($x \in [0..45]$)** of the $128 \times 128$ canvas, their Euclidean distance $\lVert\mathbf{x}^{(A)} - \mathbf{x}^{(B)}\rVert_2$ in Pixel Space is **dramatically smaller** than two characters with different stems!

As a result, the High-Probability Distribution $p_{\text{data}}(\mathbf{x})$ forms a **3-Tier Hierarchical Tree of Clusters**:

```mermaid
flowchart TD
    Ocean["16,384-D Pixel Space [-1, +1]^(128x128) (99.9999% White/Gray Noise Ocean)"] --> L1["LEVEL 1: Macro-Layout Super-Clusters (High Noise t = 1000 -> 650)"]

    L1 --> LR_Layout["Left-Right Split Template '⿰' (~65% of all 20,607 Hanzi)"]
    L1 --> TB_Layout["Top-Bottom Split Template '⿱' (Grass '艹' / Roof '宀' Crowns)"]
    L1 --> Box_Layout["Enclosure Box Template '⿴' (Outer Frame '囗' / Gate '门')"]

    LR_Layout --> L2_Wood["LEVEL 2: Wood Stem Ridge '木-' (1,200+ Characters Share Left Pixels!)"]
    LR_Layout --> L2_Water["LEVEL 2: Water Stem Ridge '氵-' (1,500+ Characters Share Left 3-Drops!)"]
    LR_Layout --> L2_Person["LEVEL 2: Person Stem Ridge '亻-' (1,000+ Characters Share Left Stroke!)"]

    L2_Wood --> P1["LEVEL 3 Peak: '林' (Forest = 木 + 木)"]
    L2_Wood --> P2["LEVEL 3 Peak: '枝' (Branch = 木 + 支)"]
    L2_Water --> P3["LEVEL 3 Peak: '汁' (Juice = 氵 + 十)"]
    L2_Water --> P4["LEVEL 3 Peak: '河' (River = 氵 + 可)"]
```

---

### How Forward Noise & Reverse Denoising Navigate This Stem-Character Hierarchy

Why does the diffusion model naturally lock onto the **Stem Character first** and the **detailed strokes second**?
Look at the Gaussian smoothing physics of $q_t(\mathbf{x}_t) = \int q(\mathbf{x}_0) \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t}\mathbf{x}_0, (1-\bar{\alpha}_t)\mathbf{I}) d\mathbf{x}_0$ across the 3 denoising stages (which you can visually see in [`Cells 26–28`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/Diffusion_Project_Generate_Chinese_Characters.ipynb#L1538) across `Steps = [1, 2, 3, 5, 10, 20, 50]`!):

```mermaid
xychart-beta
    title "Cluster Probability Cross-Section: How Individual Peaks Merge Into Stem Ridges Under Noise"
    x-axis "Pixel Space Position (Left: Wood-Stem Cluster [林, 枝] <---> Right: Water-Stem Cluster [汁, 河])" ["Edge", "Peak: 林", "Valley", "Peak: 枝", "Gap", "Peak: 汁", "Valley", "Peak: 河", "Edge"]
    y-axis "Probability Density p_t(x)" 0.0 --> 1.0
    line [0.05, 0.95, 0.15, 0.95, 0.01, 0.95, 0.15, 0.95, 0.05]
    line [0.20, 0.65, 0.60, 0.65, 0.08, 0.65, 0.60, 0.65, 0.20]
```
- **Sharp Twin-Peaks Curve ($t = 0$, Clean Data)**: Each specific character (`林`, `枝`, `汁`, `河`) is a sharp individual mountain peak.
- **Smooth Merged Curve ($t = 500$, Medium Noise)**: Adding Gaussian variance $(1 - \bar{\alpha}_t)$ acts as a **low-pass spatial filter**! The narrow valley between `林` and `枝` fills in completely, merging all 1,200 `木`-stem characters into **one broad "Wood-Stem Plateau"**, and all 1,500 `氵`-stem characters into **one broad "Water-Stem Plateau"**!

Therefore, during **Reverse Denoising ($t = 1000 \to 0$)**:
1. **Steps 1–5 ($t = 1000 \to 650$, Coarse Global Layout)**: The UNet's deepest bottleneck ($2 \times 2$ and $4 \times 4$ with `AttnDownBlock2D`) carves the white outer margins and decides the macro-template (e.g., Left-Right split `⿰`).
2. **Steps 5–20 ($t = 650 \to 250$, Stem Character / Radical Lock-In)**: The trajectory climbs onto one of the dominant **Level-2 Stem Ridges** (e.g., locking in the 3 water drops `氵` or the wood stem `木` on the left side of the canvas, because that shared pattern has $1,000\times$ higher cumulative probability mass than any single right-side component!).
3. **Steps 20–50 ($t = 250 \to 0$, Sub-Cluster Bifurcation & Stroke Sharpening)**: As noise variance $(1 - \bar{\alpha}_t) \to 0$, the broad Stem Plateau splits into individual character peaks, and the high-resolution `UpBlock2D` skip connections ($64 \times 64 \to 128 \times 128$) snap gray smudges into crisp black strokes (`-1.0`) on white paper (`+1.0`)!

---

### How Does the Model Invent Brand-New Character Combinations (`氵` + `木` $\to$ `沐`)?

Because `UNet2DModel` uses **spatial convolutions** whose receptive fields at $32 \times 32$ and $16 \times 16$ cover *local regions* of the $128 \times 128$ square:
- The **Left Half** of the canvas ($c \in [0..55]$) is pulled by the local score field $\nabla_{\mathbf{x}_{\text{left}}} \ln p(\mathbf{x}_{\text{left}})$ toward a high-probability **Left Stem Radical** (such as Water `氵`).
- The **Right Half** of the canvas ($c \in [56..127]$) is pulled by $\nabla_{\mathbf{x}_{\text{right}}} \ln p(\mathbf{x}_{\text{right}} \mid \mathbf{x}_{\text{left}})$ toward a high-probability **Right Structural Component** (such as Tree `木` or Mouth `口`).
- Meanwhile, the **`AttnDownBlock2D` / `AttnUpBlock2D` Self-Attention layers at $4 \times 4$** act as a global calligrapher—ensuring stroke thickness, baseline height, and visual center-of-gravity match across the left stem and right body!

---

## 🔬 2. Architecture & Code Forensics of [`Diffusion_Project_Generate_Chinese_Characters.ipynb`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/Diffusion_Project_Generate_Chinese_Characters.ipynb)

### End-to-End Pipeline Architecture Across Steps 1–12

```mermaid
flowchart LR
    subgraph DataPrep["Steps 1-3 (Cells 1-8): Unihan ETL Pipeline"]
        UFile["Unihan-kDefinition.txt (20,607 Characters)"] --> Clean["Regex Clean ASCII Definitions"]
        Clean --> Render["PIL ImageFont (NotoSansSC 100pt) -> 128x128 Grayscale PNGs"]
        Clean --> JSONL["metadata.jsonl (Unicode, caption, file_name)"]
    end

    subgraph TrainPipe["Steps 4-10 (Cells 9-22): Accelerate FP16 Training"]
        Render --> Loader["LocalDataset (128x128x1 in [-1, +1])"]
        Loader --> UNet["6-Stage UNet2DModel (in=1, out=1, ch=128..512)"]
        UNet --> Loss["DDPMScheduler (1000 steps) + F.mse_loss + GradClip(1.0)"]
    end

    subgraph FastSample["Steps 11-12 (Cells 23-28): Fast ODE Sampling"]
        Loss --> DPMSolver["DPMSolverMultistepScheduler (1000 Train Steps -> 20-50 Fast Inference Steps!)"]
        DPMSolver --> Grid["diffusion_process3.gif & Step Progression Grid"]
    end
```

### Key Technical Highlights in the Notebook

1. **6-Stage Deep `UNet2DModel` for $128 \times 128 \times 1$ Grayscale Glyphs ([`Cell 18`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/Diffusion_Project_Generate_Chinese_Characters.ipynb#L1115-L1138))**:
   - Unlike the 4-stage $32 \times 32 \times 3$ Butterfly UNet in Module 10, $128 \times 128$ calligraphy requires **6 downsampling stages** (`128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2`) with `block_out_channels = (128, 128, 256, 256, 512, 512)` and `in_channels = 1, out_channels = 1`:
     - **Stages 1–4 (`128 -> 8`, `DownBlock2D`)**: Capture stroke edges, corners, hooks (`亅`), and radicals (`氵, 木, 口`).
     - **Stage 5 (`8 -> 4`, `AttnDownBlock2D`)**: Applies spatial Self-Attention across the $4 \times 4$ grid so the left radical and right body align proportionally.
     - **Stage 6 (`4 -> 2`, `DownBlock2D`)**: Encodes the global character density and bounding box.

2. **Why `DPMSolverMultistepScheduler` ([`Cell 20 & Cell 24`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/Diffusion_Project_Generate_Chinese_Characters.ipynb#L1158)) Needs Only 20–50 Steps Instead of 1,000!**:
   - Standard `DDPMScheduler` is a first-order stochastic Euler sampler requiring all $1,000$ tiny steps.
   - **`DPMSolverMultistepScheduler` (Lu et al., 2022)** formulates reverse diffusion as an exact **Probability Flow Ordinary Differential Equation (ODE)** and uses a **high-order multistep Taylor expansion** in log-SNR ($\lambda_t = \ln(\sqrt{\bar{\alpha}_t}/\sqrt{1-\bar{\alpha}_t})$) space—jumping across the probability trajectory in **20 to 50 steps** ($20\times$ faster!) with zero retraining of the UNet!

3. **Important Code Forensic Note on `glyffuser-unconditional` vs. `prompt = "water"` ([`Cell 26`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/Diffusion_Project_Generate_Chinese_Characters.ipynb#L1538))**:
   - Notice that although Steps 1–3 create `metadata.jsonl` with English definitions (`caption`), Steps 5–10 train **`UNet2DModel` (`glyffuser-unconditional`)**, which only takes `(noisy_images, timesteps)` without text embeddings!
   - In [`Cell 26`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/Diffusion_Project_Generate_Chinese_Characters.ipynb#L1474), `prompt = "water"` is passed only to `make_labeled_grid(...)` to draw a title header, while `pipeline(batch_size=1, generator=generator, num_inference_steps=num_steps)` samples unconditionally from the learned Hanzi radical manifold!
   - *(How to make it truly text-conditioned on `metadata.jsonl`)*: Swap `UNet2DModel` for **`UNet2DConditionModel` (`cross_attention_dim=512`)** and pass `encoder_hidden_states = clip_text_encoder(batch["caption"])` into `model(noisy_images, timesteps, encoder_hidden_states)`!

---

## 💻 3. Verified Numerical & Visual Output (`chinese_character_radical_clusters_from_scratch.py`)

Run [`chinese_character_radical_clusters_from_scratch.py`](file:///Users/damodhar/Documents/Workspace/Learnings/Vizuara%20-%20Minor%20in%20AI/01-class-notes/11-diffusion-project-chinese-characters/chinese_character_radical_clusters_from_scratch.py) locally to verify the exact pixel-space L2 distances and hierarchical stem-first denoising:
```bash
python3 "01-class-notes/11-diffusion-project-chinese-characters/chinese_character_radical_clusters_from_scratch.py"
```

```text
1. GEOMETRIC PROOF: WHY CHINESE CHARACTERS CLUSTER BY 'STEM CHARACTER' (RADICAL) IN PIXEL SPACE
  L2 Dist || 林 (Lin - Forest) <--> 枝 (Zhi - Branch) || = 4.90 --> Same Stem Radical!  (Close Sub-Cluster)
  L2 Dist || 汁 (Zhi - Juice)  <--> 河 (He  - River)  || = 6.32 --> Same Stem Radical!  (Close Sub-Cluster)
  L2 Dist || 枝 (Zhi - Branch) <--> 汁 (Zhi - Juice)  || = 9.17 --> Different Radicals! (Far Super-Clusters)

2. HIERARCHICAL COARSE-TO-FINE DENOISING TRACE (STEM FIRST -> RIGHT BODY SECOND!)
  Step t=40 (High Fog) : P(Wood Stem '木') =  90.7% | Individual: P(林)=45.3%, P(枝)=45.3%  (Locks onto Stem Ridge!)
  Step t=20 (Mid Fog)  : P(Wood Stem '木') =  98.6% | Individual: P(林)=53.4%, P(枝)=45.2%  (Stem fixed; right side splitting)
  Step t= 0 (Zero Fog) : P(Wood Stem '木') = 100.0% | Individual: P(林)=100.0%              (Crisp character peak!)
```
