# Model Decision

## Chosen Base Model

| Field         | Value                        |
|---------------|------------------------------|
| Model         | qwen2.5-coder:1.5b           | 
| Quant level   | q4_K_M                       |
| Format        | GGUF                         |
| Served via    | Ollama                       |

## Hardware

| Field         | Value                        |
|---------------|------------------------------|
| CPU           | Intel i5-7200U               |
| GPU           | Integrated (Intel HD 620)    |
| RAM           | 16GB                         |
| Disk free     | ~146 GB                      |

## Benchmark Results

Run with `python -m vinrei.ollama_client <model>` — same prompt on each.

| Model                  | tokens/sec | Output quality (1–5) | Notes         |
|------------------------|------------|----------------------|---------------|
| qwen2.5-coder:1.5b     |            |                      |               |
| deepseek-coder:1.3b    |            |                      |               |
| starcoder2:3b          |            |                      |               |

## Decision Rationale

<!-- Why did you pick this model over the others?
     e.g. best tokens/sec, cleanest code output, smallest memory footprint -->

## Quant Level Notes

- `q4_K_M` — smallest, fastest, slight quality loss. Recommended for CPU-only.
- `q5_K_M` — balanced. Small quality gain over q4, ~20% more RAM.
- `q8_0`   — near lossless, but heaviest. Likely too slow on this hardware.

## Next Steps

- Lock this model tag in `ollama_client.py` as `DEFAULT_MODEL`
- Delete unused models: `ollama rm <model>`
