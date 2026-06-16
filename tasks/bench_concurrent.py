"""Concurrent long-context agentic perf: N sessions x R rounds, with the MTP drafter.

Simulates N concurrent agents, each holding a long context, doing R rounds of
short "checks". Round 1 = cold (N concurrent prefills); rounds 2+ reuse each
session's cached prefix (warm). llm.chat() over all N conversations lets the
scheduler run up to the admission limit concurrently and queue the rest -- so
round wall-time directly reflects how many sessions fit at once.

Reports per config (env KVDT, SLIDING, N): fits?, cold round wall-time +
aggregate tok/s, warm round wall-time + aggregate tok/s, MTP acceptance, and a
correctness check (each session retrieves ITS OWN secret -> no KV contamination).

  KVDT=turboquant_4bit_nc SLIDING=1 N=6 .venv/bin/python tasks/bench_concurrent.py
  KVDT=auto               SLIDING=0 N=6 .venv/bin/python tasks/bench_concurrent.py
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
TARGET_TOK = int(os.environ.get("TARGET_TOK", "12000"))
N = int(os.environ.get("N", "6"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))
NGEN = int(os.environ.get("NGEN", "48"))

print(f"### concurrent KVDT={KVDT} sliding={os.environ['VLLM_TQ_SLIDING_WINDOW']} "
      f"N={N} rounds={ROUNDS} init~{TARGET_TOK}tok/sess drafter=ON ###", flush=True)

kw = dict(
    model=MODEL,
    speculative_config={"model": DRAFT, "num_speculative_tokens": 1},
    max_model_len=CTX,
    max_num_seqs=N,
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
    print(f"DOES NOT FIT (KVDT={KVDT}, N={N}): {type(e).__name__}: {msg}", flush=True)
    print("CONC DONE", flush=True)
    raise SystemExit(0)
print(f"[LOADED in {time.time()-t0:.0f}s]", flush=True)

filler = "The river was calm and the market sold figs under a pale sky. "
per = max(1, len(llm.get_tokenizer().encode(filler)))
reps = max(1, TARGET_TOK // per)
words = ["falcon", "harbor", "cinder", "willow", "quartz", "meadow", "tundra", "zephyr"]
secrets = [f"azure-{words[i % len(words)]}-{i:02d}" for i in range(N)]

checks = [
    "What is the secret code in the document? Reply with ONLY the code.",
    "Does the document mention kites? yes or no.",
    "In one short sentence, what do the merchants sell?",
    "Repeat the secret code exactly.",
]
convs = [
    [{"role": "user",
      "content": (filler * reps
                  + f"\n\nIMPORTANT: the secret code is '{secrets[i]}'.\n\n"
                  + checks[0])}]
    for i in range(N)
]
sp = SamplingParams(temperature=0.0, max_tokens=NGEN)

rounds = []
for r in range(ROUNDS):
    t0 = time.time()
    outs = llm.chat(convs, sp)
    dt = time.time() - t0
    n_out = sum(len(o.outputs[0].token_ids) for o in outs)
    texts = [o.outputs[0].text.strip() for o in outs]
    rounds.append((dt, n_out, texts))
    nxt = checks[(r + 1) % len(checks)]
    for i, o in enumerate(outs):
        convs[i].append({"role": "assistant", "content": texts[i]})
        convs[i].append({"role": "user", "content": nxt})

# Correctness: round-0 asked for the secret -> each session must return its own.
hits = sum(secrets[i] in rounds[0][2][i] for i in range(N))

acc = drf = 0
for m in llm.get_metrics():
    nm = m.name.rstrip("_total")
    if nm.endswith("spec_decode_num_accepted_tokens"):
        acc = m.value
    if nm.endswith("spec_decode_num_draft_tokens"):
        drf = m.value
rate = (acc / drf) if drf else float("nan")
f, tot = torch.cuda.mem_get_info()

print(f"RESULT KVDT={KVDT} sliding={os.environ['VLLM_TQ_SLIDING_WINDOW']} N={N}", flush=True)
cold = rounds[0]
print(f"  COLD round 1 : {cold[0]:.1f}s for {N} sessions, {cold[1]} tok -> "
      f"{cold[1]/cold[0]:.1f} tok/s aggregate", flush=True)
warm = rounds[1:]
if warm:
    wt = sum(x[0] for x in warm) / len(warm)
    wtps = sum(x[1] for x in warm) / sum(x[0] for x in warm)
    print(f"  WARM rounds 2..{ROUNDS} (prefix-cached): avg {wt:.2f}s for {N} "
          f"concurrent turns -> {wtps:.1f} tok/s aggregate", flush=True)
print(f"  secret retrieval (round 1): {hits}/{N} correct (no cross-session leak)", flush=True)
print(f"  mtp_accept_rate = {rate:.3f}", flush=True)
print(f"  GPU = {(tot-f)/2**30:.2f}/{tot/2**30:.2f} GiB", flush=True)
print("CONC DONE", flush=True)
