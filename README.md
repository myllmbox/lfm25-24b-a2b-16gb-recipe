# LFM2.5-24B-A2B on one 16 GB GPU

[LFM2.5-24B-A2B-DERISKED-NVFP4](https://huggingface.co/Blackfrost-AI/LFM2.5-24B-A2B-DERISKED-NVFP4) (Blackfrost-AI's
LFM2.5-24B-A2B derivative, 24B MoE, ~2B active, NVFP4) on a single RTX 50-series card with 16 GB.
**221 tok/s on one stream, 552 tok/s at 4 streams, 131k context, a 156k-token KV pool.** One command.

**Side by side with every other stack:** [myllmbox.com](https://myllmbox.com)

## Measured (RTX 5080 16 GB, image v1, the shipped `recipe.yaml`)

Mixed prompt (code + explanation), thinking on. Aggregate tok/s of all streams, averages of 3 runs, each measured until
the first stream finishes; **peak** = the best 5-s window.

| | c=1 | c=2 | c=3 | c=4 |
|---|---|---|---|---|
| average | **221** | **348** | **456** | **552** |
| peak | 225 | 355 | 469 | 571 |
| per stream | 221 | 174 | 152 | 138 |

### Long context (one request)

Decode = 1,000 tokens after the prompt. Hot = the next turn of the same conversation (prefix cache).

| prompt tokens | first token | hot first token | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| 1k | **0.15 s** | 0.06 s | 8,233 | **225** |
| 4k | 0.28 s | 0.07 s | 15,159 | 223 |
| 16k | 0.82 s | 0.09 s | **20,153** | 212 |
| 32k | 1.9 s | 0.11 s | 17,157 | 200 |
| 64k | 5.5 s | 0.16 s | 12,004 | 179 |
| 96k | 10.9 s | 0.22 s | 9,071 | 162 |
| 126k | 17.5 s | **0.26 s** | 7,370 | 149 |

## Quick start

```bash
git clone https://github.com/myllmbox/lfm25-24b-a2b-16gb-recipe.git
cd lfm25-24b-a2b-16gb-recipe
./run.sh        # first run downloads ~14G, then serves on :8000
```

`./stop.sh` stops it. `./view.sh` shows the live log and GPU memory. Requirements: Linux, docker + the NVIDIA container
toolkit, driver 590 or newer, a Blackwell GPU (RTX 50-series) with 16 GB that nothing else uses — the checkpoint is NVFP4
W4A4 and does not run on older cards. Boot to first answer: about a minute.

**All configuration lives in [`recipe.yaml`](recipe.yaml)** — image, weights, port, KV budget, every vLLM flag.

```bash
curl http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "LFM2.5-24B-A2B-DERISKED-NVFP4",
  "messages": [{"role": "user", "content": "hello"}]
}'
```

The model always thinks; the reasoning arrives as `reasoning_content`. Tool calls work (`tools` in the request).

## Tuning (recipe.yaml)

- **Memory.** The serve uses ~15.5 of 16 GB. `kv-cache-memory: "1600000000"` = 156,125 tokens. If the card also drives a
  display, or a boot runs out of memory, lower it (and `min_free_vram_mib` by the same amount). A 24/32 GB card can take
  several GB more.
- **Context.** `max-model-len: 131072`, the model's maximum. The pool is shared by all requests.
- **Seats.** `max-num-seqs: 4`. The `cudagraph_capture_sizes` list in `compilation-config` must cover every seat count;
  raise both together.
- **Sampling.** `override-generation-config` keeps the publisher's temperature 0.1: near-greedy, poor variety. Set 0.6-0.7
  for more varied answers. A request's own `temperature` always wins.

## What's in the image

`myllmbox/lfm25-24b-a2b-vllm:v1` — `vllm/vllm-openai:v0.31.0` (the checkpoint's validated runtime) plus our loader
(`load-format: mbx`): the weights load straight to the GPU without filling the page cache.

Digest: `sha256:0c5448641a035c920942a3197074e541ac5c9ea82a67914eaa90514c5d93d4d4`

## The full box

The same model also runs under [myllmbox](https://myllmbox.com) with a public HTTPS tunnel, dashboard, keepalive and
multi-model management — same image, same weights.

## License

Weights: [LFM Open License v1.0](https://huggingface.co/Blackfrost-AI/LFM2.5-24B-A2B-DERISKED-NVFP4/blob/main/LICENSE)
(Liquid AI; read it before commercial use). Kit scripts: MIT.
