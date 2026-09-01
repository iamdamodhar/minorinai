# Business Case & Revenue Model: Generative AI on Myntra Web

---

## 1. Executive Summary & Financial Impact

Transforming Myntra.com with Generative AI is not merely a user experience upgrade; it is a massive driver of top-line Gross Merchandise Value (GMV) expansion and bottom-line EBITDA profitability.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        CORE FINANCIAL DRIVERS                          │
├────────────────────────────────┬───────────────────────────────────────┤
│ 1. Conversion Rate (CR) Uplift │ 2.2% Baseline ──► 3.5% (+59% Uplift)  │
├────────────────────────────────┼───────────────────────────────────────┤
│ 2. Average Order Value (AOV)   │ ₹1,850 ──► ₹2,200 (+19% via Bundles)  │
├────────────────────────────────┼───────────────────────────────────────┤
│ 3. Return Rate Reduction (RTO) │ 32% ──► 24% (-800 bps Reduction)      │
├────────────────────────────────┼───────────────────────────────────────┤
│ 4. Annual Net EBITDA Addition  │ ₹240+ Crore direct bottom-line gain   │
└────────────────────────────────┴───────────────────────────────────────┘
```

---

## 2. Quantitative Revenue Modeling

### A. Conversion Rate (CR) Uplift
- **Problem**: Current keyword search and 15-filter grids cause high drop-off. An average session is 18 minutes with only ~2.2% converting.
- **GenAI Impact**:
  - Conversational Stylist ("Maya 2.0") and Multimodal Vibe Search reduce search time from 18 minutes to <3 minutes.
  - Interactive outfit bundling delivers pre-matched items matching customer intent.
- **Model**:
  - Monthly Web Sessions: 50,000,000
  - Baseline Orders (2.2% CR): 1,100,000 orders/month
  - Post-GenAI Orders (3.5% CR): 1,750,000 orders/month
  - **Incremental Monthly Orders**: **+650,000 orders/month**

---

### B. Average Order Value (AOV) Expansion
- **Problem**: Most users buy single standalone garments (e.g., 1 shirt or 1 kurta), resulting in an average basket size of 1.3 items.
- **GenAI Impact**:
  - GenAI "Complete the Look" automatically presents matching footwear, accessories, and outerwear with a 1-click bundle discount (e.g., 5% off when buying the complete look).
  - Multi-item basket size increases from **1.3 items $\rightarrow$ 1.8 items**.
- **Model**:
  - Baseline AOV: ₹1,850
  - Post-GenAI AOV: ₹2,200 (+₹350 per order)

---

### C. The Reverse Logistics & Return Rate Fix (EBITDA Multiplier)
- **Problem**: In Indian fashion e-commerce, **30%–35% of all orders are returned**.
  - 60% of returns are driven by **sizing & fit doubt** (e.g., ordering two sizes to keep one).
  - 25% are driven by **visual mismatch** ("looks different from the model image").
  - Every return costs **₹180–₹250** in reverse logistics, return inspection, repackaging, and inventory lockup.
- **GenAI Solution**:
  - Photorealistic Diffusion Virtual Try-On previews garment drape on real body shapes.
  - AI Fit & Size Synthesizer provides real-world sizing advice ("Order 1 size up for broad shoulders").
- **Cost Savings Calculation**:
  - Annual Delivered Orders: 25,000,000
  - Baseline Returns (32%): 8,000,000 returned items
  - Post-GenAI Returns (24%): 6,000,000 returned items
  - **Avoided Returns**: **2,000,000 returns/year**
  - Cost per Return Saved: ₹180
  - **Direct Annual Logistics Savings**: $2,000,000 \times \text{₹180} = \mathbf{₹360,000,000\text{ (₹36 Crore)}}$

---

## 3. Cost-to-Serve & Cloud GPU Optimization

To ensure profitability, the inference cost per session must remain negligible ($<\text{₹0.10}$ per session).

```
Cost Mitigation Matrix:
├── 1. Semantic Caching (>70% hit rate): Zero GPU inference cost for cached intents.
├── 2. Quantized SLMs (Gemma 2B / Llama 3.2 3B): Runs on cost-effective NVIDIA L4 GPUs (~$0.70/hr).
├── 3. Batch Pre-computation: Heavy outfit graphs generated offline during off-peak cloud discount hours.
└── 4. Blended Inference Cost: Projected at ₹0.06 per user session (<0.003% of AOV).
```

---

## 4. Summary: The 3-Year Strategic Moat

| Strategic Dimension | Without GenAI (Status Quo) | With GenAI Transformation |
| :--- | :--- | :--- |
| **Search Paradigm** | Keyword matching & checkbox filters | Semantic, visual & contextual reasoning |
| **User Role** | Manual catalog browser | Guided client with a personal fashion stylist |
| **Purchase Basket** | Isolated single-SKU purchases | Cohesive head-to-toe curated looks |
| **Returns Risk** | High return rate (32%) | Low return rate (24%) via Virtual Try-On |
| **Channel Reach** | Limited to Myntra website & app | Omnipresent in ChatGPT, Gemini & WhatsApp |
