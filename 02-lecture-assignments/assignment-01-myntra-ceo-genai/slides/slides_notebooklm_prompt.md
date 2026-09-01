# NotebookLM & AI Presentation Generation Prompt

Use this document to paste directly into **Google NotebookLM**, **Gamma.app**, **Beautiful.ai**, or **Microsoft 365 Copilot** to instantly generate your visual presentation deck or audio overview.

---

## 📋 Copy-Paste Prompt for NotebookLM / Gamma

```text
Act as a world-class presentation designer and executive speechwriter. Based on the source notes provided below, generate an inspiring, highly professional 9-slide visual presentation deck from the perspective of the CEO of Myntra addressing the engineering and web development team. 

Key Tone & Requirements:
1. Executive, visionary, encouraging, yet technically grounded in high-scale web engineering (60M users, 200+ orders/sec, sub-second latency).
2. Use creative, memorable analogies (e.g. "The 10,000-page phonebook vs. The Luxury Boutique Stylist", "The Formula 1 Pitstop Architecture", "The Magic Dressing Mirror").
3. Emphasize both customer experience (frictionless discovery, zero-click headless shopping) and core business revenue levers (Conversion rate boost, AOV expansion, slashing 30% return rates).
4. Outline concrete architectural patterns (Semantic Caching, Tiered SLM/LLM, SSE Token Streaming, Async Diffusion).

---
SOURCE CONTENT:
[Insert the content from slides_content.md below]
```

---

## 📄 Complete Source Content for NotebookLM

```markdown
# Myntra Generative AI Web Transformation

### Executive Context:
- Speaker: Chief Executive Officer, Myntra
- Audience: Web Developers, Product Managers, Platform Engineers
- Company Context: India's largest fashion e-commerce platform with 60M+ active users, 10M+ products, and peak sales handling 200+ transactions per second (TPS).

### Problem Statement:
1. Current search paradigm is a "Digital Warehouse": users must manually apply 15 filters (color, size, brand, price) and scroll through 200+ thumbnails, leading to decision fatigue (18-minute session before cart abandonment).
2. The 30% Return Problem: Fashion e-commerce suffers from 25-35% return rates due to sizing uncertainty and visual mismatch, costing ₹150-₹250 per return in reverse logistics.
3. High Latency Threat: Traditional LLMs take 2-5 seconds to generate responses, which would destroy user experience during high-traffic festive sales.

### The GenAI Solution - 4 Core Pillars on Myntra Web:
1. Conversational Stylist ("Maya 2.0"): Natural language understanding to assemble complete head-to-toe outfits for specific occasions, budgets, and climates in 1 click.
2. Photorealistic AI Virtual Try-On ("The Magic Mirror"): Diffusion-based neural draping on custom 3D avatars or user selfies to preview real-world fit and fabric drape.
3. Multimodal Vibe Search: Translating social media screenshots or abstract aesthetic prompts ("90s Mumbai Monsoon Retro") into exact catalog matches.
4. AI Review & Fit Synthesizer: Summarizing thousands of customer reviews into 3 actionable bullet points on sizing, fabric texture, and longevity.

### Headless Conversational Commerce (The Target Model):
- Integrating Myntra directly into conversational platforms (ChatGPT, Google Gemini, WhatsApp).
- Customers planning vacations or events in ChatGPT can have Myntra carts built and checked out in 1 click without leaving the conversation.

### High-Scale Architecture (200+ TPS & <200ms Latency):
- Semantic Caching (Redis + Vector DB): Caches 75% of recurring festive fashion queries, serving responses in <15ms.
- Multi-Tier Model Routing: 80% of fast queries routed to Small Language Models (SLMs) in <50ms; complex multi-item queries routed to heavy LLMs.
- Server-Sent Events (SSE) Token Streaming: First token delivered in <200ms for instant perceived responsiveness.
- Pre-Generated Pipelines: Batch pre-computing review summaries and outfit matrices during off-peak hours.

### Business Economics & Revenue Impact:
- Conversion Rate: Increases from 2.5% to 3.8% (+52% relative uplift).
- Average Order Value (AOV): Increases by +18% through automated multi-item complete-the-look bundles.
- Return Reductions: Slashes return rates from 30% down to 22%, saving over ₹180 Crore annually in reverse logistics.
```
