"""Agentic long-context perf: ONE big prefill, then repeated WARM turns.

Simulates a long agentic workflow: load a long document once (cold prefill),
then issue several short follow-up "checks" that reuse the cached prefix
(prefix caching), so each warm turn only processes the new tokens + decode.
This is the metric that matters for agent loops -- the cold prefill is a
one-time cost.

Reports per config (env KVDT, SLIDING): fits?, cold turn-1 latency, warm
per-turn latency (turns 2..N, prefix-cached), decode tok/s, MTP acceptance.

  KVDT=turboquant_4bit_nc SLIDING=1 .venv/bin/python tasks/bench_agentic.py
  KVDT=auto               SLIDING=0 .venv/bin/python tasks/bench_agentic.py
"""
import os

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_TQ_SLIDING_WINDOW", os.environ.get("SLIDING", "0"))
import time

import torch
from vllm import LLM, SamplingParams

MODEL = "/home/lucas/ai/models/google/gemma-4-12B-it-qat-w4a16-ct"
DRAFT = "/home/lucas/ai/models/gemma4-mtp-assistant-st"
KVDT = os.environ.get("KVDT", "turboquant_4bit_nc")
CTX = int(os.environ.get("CTX", "200000"))
TARGET_TOK = int(os.environ.get("TARGET_TOK", "120000"))
NGEN = int(os.environ.get("NGEN", "48"))

print(f"### agentic KVDT={KVDT} sliding={os.environ['VLLM_TQ_SLIDING_WINDOW']} "
      f"ctx={CTX} init~{TARGET_TOK}tok drafter=ON ###", flush=True)

kw = dict(
    model=MODEL,
    speculative_config={"model": DRAFT, "num_speculative_tokens": 1},
    max_model_len=CTX,
    max_num_batched_tokens=2560,
    gpu_memory_utilization=0.85,
    enforce_eager=True,
    trust_remote_code=True,
    disable_log_stats=False,
)
if KVDT != "auto":
    kw["kv_cache_dtype"] = KVDT

t0 = time.time()
try:
    llm = LLM(**kw)
except Exception as e:
    msg = str(e).splitlines()[0][:200] if str(e) else type(e).__name__
    print(f"DOES NOT FIT at ctx={CTX} (KVDT={KVDT}, drafter ON): {type(e).__name__}: {msg}", flush=True)
    print("AGENTIC DONE", flush=True)
    raise SystemExit(0)
print(f"[LOADED in {time.time()-t0:.0f}s]", flush=True)

filler = "The river was calm and the market sold figs under a pale sky. "
per = max(1, len(llm.get_tokenizer().encode(filler)))
reps = max(1, TARGET_TOK // per)
doc = (filler * reps
       + "\n\nIMPORTANT: the vault passphrase is 'azure-pelican-77' and the gate code is 4821.\n\n")

# Agentic follow-ups: short "reiterated checks" over the same long context.
checks = [
    "What is the vault passphrase? One short sentence.",
    "What is the gate code? Just the number.",
    "Does the document mention kites? Answer yes or no.",
    "In one sentence, what do the merchants sell?",
    "Repeat the vault passphrase exactly.",
    "Is the sky described as pale? Yes or no.",
]

conversation = [{"role": "user", "content": doc + checks[0]}]
sp = SamplingParams(temperature=0.0, max_tokens=NGEN)

n_prompt0 = None
turns = []  # (latency_s, n_out, text)
for i, _ in enumerate(checks):
    t0 = time.time()
    out = llm.chat([conversation], sp)[0]
    dt = time.time() - t0
    if n_prompt0 is None:
        n_prompt0 = len(out.prompt_token_ids)
    txt = out.outputs[0].text.strip()
    n_out = len(out.outputs[0].token_ids)
    turns.append((dt, n_out, txt))
    # grow the conversation -> next turn reuses the cached prefix.
    conversation.append({"role": "assistant", "content": txt})
    if i + 1 < len(checks):
        conversation.append({"role": "user", "content": checks[i + 1]})

acc = drf = 0
for m in llm.get_metrics():
    n = m.name.rstrip("_total")
    if n.endswith("spec_decode_num_accepted_tokens"):
        acc = m.value
    if n.endswith("spec_decode_num_draft_tokens"):
        drf = m.value
rate = (acc / drf) if drf else float("nan")
f, tot = torch.cuda.mem_get_info()

print(f"RESULT KVDT={KVDT} sliding={os.environ['VLLM_TQ_SLIDING_WINDOW']} "
      f"init_prompt_tokens={n_prompt0}", flush=True)
cold = turns[0]
print(f"  COLD turn 1 : {cold[0]:.1f}s  ({cold[1]} tok out)  "
      f"-> {cold[1] / cold[0]:.1f} tok/s incl full prefill", flush=True)
warm = turns[1:]
if warm:
    avg = sum(t[0] for t in warm) / len(warm)
    tps = sum(t[1] for t in warm) / sum(t[0] for t in warm)
    print(f"  WARM turns 2..{len(turns)} (prefix-cached): avg {avg:.2f}s/turn, "
          f"{tps:.1f} tok/s", flush=True)
    for i, (dt, n_out, txt) in enumerate(warm, start=2):
        print(f"    turn {i}: {dt:.2f}s {n_out}tok -> {txt[:60]!r}", flush=True)
print(f"  mtp_accept_rate = {rate:.3f}", flush=True)
print(f"  GPU = {(tot - f) / 2**30:.2f}/{tot / 2**30:.2f} GiB", flush=True)
print("AGENTIC DONE", flush=True)
