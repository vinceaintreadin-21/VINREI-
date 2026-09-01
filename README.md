VINREI — Local AI Coding Assistant

Build your own AI for coding. No closed models (no Gemini / OpenAI / Claude), no
paid APIs, no per-token metering. Everything runs locally through [Ollama](https://ollama.ai),
fine-tuned on top of an open-weight base model.

**Stack:** Python for tooling / agent logic, C++ only where hot paths demand it.

---

## Philosophy

- **Own everything that matters to you.** We build on open weights, not closed APIs.
- **Local first.** Sent your source code no further than your own RAM.
- **Fine-tune, don't reinvent.** Start from a strong open-weight base model and adapt
  it to coding tasks. Training *everything* from scratch is a multi-GPU-month investment
  you almost never need.
- **Ollama is the runtime**, not the model. Ollama serves models and provides an
  OpenAI-compatible API; the intelligence comes from the weights we ship into it.

---

## Architecture overview

```
┌──────────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│   Agent (Python)     │ ──▶ │   Ollama server   │ ──▶ │  Fine-tuned model  │
│  planning/tool loop  │ ◀── │  OpenAI-compat    │ ◀── │  (open weights)    │
│  CLI / TUI / server  │     │  API, /api/*      │     │                    │
└──────────────────────┘     └──────────────────┘     └────────────────────┘
        │      │                    │                          ▲
        │      ▼                    ▼                          │
   ┌──────────────┐        ┌──────────────┐        ┌───────────────┐
   │ Repo context │        │  Vector DB   │        │ Train/val     │
   │ (tree, grep, │        │ (embeddings) │        │ data pipeline │
   │ git diff)    │        │              │        │ (code -> data)│
   └──────────────┘        └──────────────┘        └───────────────┘
```

Optional C++ layer: tokenization, prompt assembly, retrieval, and inference glue
that must be fast. Wrapped with `pybind11` / `ctypes` and called from Python.

---

## Roadmap

### Phase 0 — Foundations (week 1–2)

- [/] Initialize repo: Python 3.11+, `pyproject.toml`, ruff + mypy + pytest.
- [/] Install and pin **Ollama**; learn the CLI (`ollama pull`, `serve`, `list`).
- [/] Pull candidate open-weight code bases: `qwen2.5-coder`, `deepseek-coder`,
      `codellama`, `starcoder2`. Benchmark raw quality on your own tasks.
- [/] Write a thin `ollama_client.py` that talks to the OpenAI-compatible API
      (`http://localhost:11434/v1`) — no external SDKs. **No token counting.**
- [/] Smoke test: stream a completion, measure *tokens/sec* on your hardware
      (this decides which model size you can run: 1–3B, 7–8B, 14B, 30B+).
- [/] Decide base model + quant level (GGUF `q4_K_M` / `q5_K_M` / `q8_0`) and
      lock it in a `MODEL.md`.

### Phase 1 — Coding assistant v0 (week 3–6)

A working assistant before any training happens. Everything below works with the
raw base model; fine-tuning only makes it better.

- [/] **CLI chat** (`vinrei "explain this bug"`), streaming output, plain text v1.
- [/] **Repo grounding:** codebase tree walking, `grep`/`rg` search, file reading
      with size caps, `.gitignore` awareness.
- [/] **RAG** on your own code: chunking + embeddings (`all-MiniLM`-class or the
      model's own embeddings), similarity search, inject top-k chunks into context.
- [/] **System + task prompts:** role framing, "only edit files when asked",
      constrained output formats (fenced code blocks / diff).
- [/] **Tool loop / agent:** model can request `read`, `grep`, `edit`, `run`.
      Implement tool-call parsing and execution (JSON or function-calling).
- [skip to phase 5] **C++ perf experiments:** if prompt assembly or embedding retrieval is the
      bottleneck, extract it into a small C++ core with `pybind11`.
- [/] Suggest/complete inline: FIM (fill-in-the-middle) via Ollama's `/api/generate`.

**Exit criteria:** the assistant can explain, search, and edit a real repo
end-to-end on your machine.

### Phase 2 — Training data pipeline (week 7–10)

The data is the model. Most of the quality lives here.

- [/] Collect coding instruction data from your own sessions: prompts, answers,
      and (critically) the diffs you actually accepted.
- [/] Curate open datasets: The Stack (v2), CodeAlpaca, Magicoder-Evol, etc.
      Filter for licenses, deduplicate, drop garbage (min length, no PII).
- [/] Normalize to a single fine-tune format: `<prompt>, <golden completion>`.
- [/] Compare instruction tuning (SFT) vs. preference tuning (DPO) _later_; start
      with SFT.
- [/] Repo-level examples: each sample = task + repo context window (tree + files)
      + expected edit. This teaches "coding agent" behavior, not just text.
- [/] Store dataset versioned (parquet + hash), track splits (train/val/test),
      add a smoke eval harness (human_eval or a 20-task homemade suite).

**Exit criteria:** the dataset covers chat, codegen, and agentic edit tasks, is
versioned, and reproduces a baseline score.

### Phase 3 — Fine-tuning (week 11–14)

Weights trained locally or on rented GPUs — still no per-token API costs.

- [/] **LoRA / QLoRA** fine-tune with a descendent of HuggingFace transformers or
      PEFT; `bitsandbytes` 4-bit base for consumer GPUs.
- [/] Track with **wandb / mlflow**: loss, eval perplexity, eval-task pass rate.
- [/] Style/format overfitting check: keep validation data out of the train set
      (data leakage is the #1 silent killer).
- [/] Full fine-tune vs LoRA: LoRA first; escalate only if quality stalls.
- [/] Checkpoint recipes: merge LoRA → full weights → quantize to GGUF
      (`llama.cpp` / `mlx` / `torch` → `ollama create`).
- [/] Load into Ollama as a custom model and run the Phase 1 harness
      **unchanged** — prove the fine-tune made the agent measurably better.

**Exit criteria:** custom model runs under Ollama and beats the base on your eval
suite; deltas logged and reproducible.

### Phase 4 — Preference tuning & polish (week 15–18)

- [ ] Generate candidate answers per prompt, rank them, build a preference set.
- [ ] DPO / ORPO to teach "do the right edit" vs. "plausible-looking edit".
- [ ] Prompt-injection and harm checks: don't run rm -rf because the model said so.
- [ ] Guardrails: sandboxed command execution (container / seccomp) for the agent loop.

### Phase 5 — Performance & productization (week 19+)

- [ ] Hardware profiling: prompt prefill vs. decode speed, KV cache size,
      context window (e.g., 4k → 32k with RoPE scaling).
- [ ] Speculative decoding (draft model + verifier) for 2–4× latency win.
- [ ] C++ hot paths: tokenizer, prompt cache, and the agent's diff application tool.
- [ ] Interactive TUI (rich / textual) and optional HTTP server for IDE plugins.
- [ ] Packaging: single wheel, `docker` image with bundled model, offline-first docs.

---

## Suggested tech

| Concern            | Choice                                                    |
|--------------------|-----------------------------------------------------------|
| Runtime / serving  | Ollama, OpenAI-compatible `/v1` API                       |
| Model base         | qwen2.5-coder / deepseek-coder / starcoder2 (open weights)|
| Fine-tuning        | PEFT + bitsandbytes (LoRA/QLoRA), transformers            |
| Quantization       | GGUF (q4_K_M … q8_0) via llama.cpp                        |
| Experiment tracking| wandb or mlflow                                           |
| Datasets           | The Stack v2, CodeAlpaca, Magicoder-Evol + **your data**  |
| Language           | Python (agent, tooling); C++/pybind11 (hot paths)         |
| Eval              | human_eval, MBPP + homemade repo-level suite              |

---

## Hardware notes

- **LLM inference is memory-bound.** A 7–8B model at Q4 needs ~4–6 GB VRAM and
  roughly 40–60 tokens/sec on a mid-range GPU.
- No GPU? GGUF CPUs work but aim for 1–3B models, or use partial offload.
- Training is the expensive part: QLoRA 7B can fit on 12 GB; full fine-tunes want
  24+ GB or rented A100/H100.

---

## Success metrics

- **Local first:** zero API requests, zero token metering, fully offline capable.
- **Quality:** your eval suite pass rate improves after every training round.
- **Latency:** streaming first token < 1 s, sustained speed within 2× of a
  mid-range commercial model *at any price*.
- **Openness:** every artifact — data, configs, weights, code — is reproducible
  from this repo.

---

## Quick start (current state: groundwork)

```bash
# ollama server
ollama serve &

# interactive eval of a base model, no training needed yet
python -m vinrei.chat --model qwen2.5-coder:7b
```

---

## Milestones in one line

0. ship Gemini-clone-grade **assistant** using a raw open model → 1. add agentic
   tools → 2. build a **dataset** from your real usage → 3. **fine-tune** your own
   weights → 4. preference-tune → 5. make it **fast** with C++.