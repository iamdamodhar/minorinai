# Technical Architecture: High-Scale, Sub-Second GenAI for Myntra Web
**Target Performance**: 200+ Orders/Sec (TPS), 50,000+ Requests/Sec (RPS), <200ms Time-to-First-Token (TTFT)

---

## 1. System Architecture Diagram

```
[ Web Client / Mobile Web ]
            │ (HTTP/2 / WebSocket / SSE)
            ▼
[ Cloudflare Edge / CDN & WAF ]
            │
            ▼
[ API Gateway (Envoy / Kong) ]
            │
    ┌───────┴───────────────────────────────┐
    ▼                                       ▼
[ Semantic Cache Layer ]             [ Fast Auth & Rate Limiter ]
 (Redis Cluster + Milvus)
    │
    ├─► Cache Hit (>70% overlap) ──► Instant Response (<15ms)
    │
    └─► Cache Miss (<30%)
            │
            ▼
[ Intelligent Model Router & Orchestrator ]
            │
    ┌───────┼────────────────────────────────────────┐
    ▼       ▼                                        ▼
[ Tier 1: SLM Cluster ]               [ Tier 2: Heavy LLM Pool ]       [ Diffusion VTO Workers ]
(Llama-3-8B-Instruct /                (Claude 3.5 Sonnet /              (ControlNet + Virtual
 Gemma-2-9B Quantized)                 GPT-4o via Private VPC)           Try-On LoRA on A100s)
  • Intent classification (<40ms)       • Multi-step styling              • Async render queue
  • Filter & entity extraction          • Nuanced fashion reasoning       • WebGL preview on client
  • Micro-copy & review summaries
            │                                        │                                │
            └───────────────────────┬────────────────┘                                │
                                    ▼                                                 ▼
                        [ Vector Search & Catalog DB ]                  [ Cloud Storage / CDN ]
                         (Pinecone / Qdrant + ScyllaDB)                  (Rendered Try-On Assets)
                                    │
                                    ▼
                        [ Server-Sent Events (SSE) ]
                                    │
                                    ▼
                        [ Instant Client UI Render ]
```

---

## 2. The 4 Technical Pillars of Sub-Second Latency

### Pillar 1: Vector-Based Semantic Caching
- **Problem**: Querying an LLM for every single keystroke or query introduces 1,000–3,000ms latency and ballooning GPU costs.
- **Solution**:
  - E-commerce festive queries have massive repetition (e.g., *"red lehenga under 5000"*, *"affordable bridal lehenga red color"*).
  - Queries are embedded using a lightweight embedding model (e.g., `all-MiniLM-L6-v2` or `BGE-small` in <10ms).
  - Check cosine similarity in Redis Vector Search against past query embeddings.
  - If similarity score $> 0.93$, the pre-computed, verified styling response is returned in **<15ms**.

---

### Pillar 2: Tiered Model Routing (SLM vs. LLM)
- **Problem**: 80% of e-commerce tasks do not require a massive frontier model.
- **Solution**:
  - **Tier 1 (Small Language Models - SLMs)**: 8B parameter models running quantized (FP8 / AWQ) on NVIDIA L4 GPUs via **vLLM** or **TensorRT-LLM**. Handles query rewriting, attribute extraction, and review summarization in **<40ms**.
  - **Tier 2 (Frontier LLMs)**: Complex conversational queries (e.g., *"What should I wear to a winter wedding in Kashmir with snow?"*) are routed to high-parameter models via private cloud endpoints.

---

### Pillar 3: Token Streaming (SSE & Progressive Web Hydration)
- **Problem**: Waiting for the full JSON response from an LLM takes 2–4 seconds, causing user abandonment.
- **Solution**:
  - Connect client to server using **Server-Sent Events (SSE)**.
  - Stream the first token within **180ms**.
  - As the LLM mentions product IDs, the web client progressively fetches and hydrates product cards in real time.

---

### Pillar 4: Asynchronous & Client-Accelerated Virtual Try-On
- **Problem**: Full diffusion image synthesis takes 3–6 seconds per image.
- **Solution**:
  - **Client-Side Approximation**: Use lightweight WebGL / Three.js 3D mesh warping for instant (<100ms) interactive preview.
  - **Server-Side Photorealistic Draping**: Queue high-fidelity Diffusion requests via Redis Celery workers; stream the final 4K texture onto the client model asynchronously.

---

## 3. Headless Agentic Commerce API (Target & ChatGPT Model)

To enable zero-click discovery and checkout from external conversational AI platforms (ChatGPT, Gemini, WhatsApp assistants), Myntra exposes standardized **Model Context Protocol (MCP)** and OpenAPI endpoints:

```json
{
  "endpoint": "/api/v1/agent/curate-and-checkout",
  "method": "POST",
  "payload": {
    "user_query": "Outfit for 3-day trekking trip to Manali in October",
    "budget_inr": 8000,
    "gender": "female",
    "size_preferences": { "top": "M", "bottom": "30", "footwear": "UK 6" }
  },
  "response": {
    "look_id": "look_manali_trek_01",
    "bundle_items": [
      { "sku": "WINT-JKT-882", "name": "Windproof Fleece Trekking Jacket", "price": 3499 },
      { "sku": "TRK-PNT-104", "name": "Water-Resistant Cargo Track Pants", "price": 1999 },
      { "sku": "TRK-SHOE-55", "name": "High-Ankle Grip Trekking Boots", "price": 2499 }
    ],
    "total_bundle_price": 7497,
    "discount_applied": "5% Look Bundle Discount",
    "instant_checkout_url": "https://www.myntra.com/checkout/instant?bundle=look_manali_trek_01&token=eyJhbGci..."
  }
}
```

---

## 4. Latency & Resilience Budget

| Component | Target Latency | Failover / Fallback Strategy |
| :--- | :--- | :--- |
| **Semantic Cache Lookup** | <15 ms | Fallback to Vector Search |
| **Intent Classifier (SLM)** | <40 ms | Fallback to Keyword Regex Parser |
| **Catalog Vector Search** | <35 ms | Fallback to Elasticsearch index |
| **Time-to-First-Token (TTFT)** | <200 ms | Instant fallback to pre-computed static bundles |
| **Virtual Try-On Preview** | <100 ms (WebGL) | Shows 2D high-res model image while rendering |
