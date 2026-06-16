"""200K-context perf at the edge of the 16GB card, WITH the MTP drafter.

Compares KV dtypes (env KVDT): bf16 (auto, no turboquant) vs turboquant_4bit_nc.
Reports: load time, whether it even FITS, TTFT (prefill of a ~190K prompt),
decode tokens/s (with MTP speculative decoding), and draft acceptance.

TTFT vs decode are separated with the prefix-cache trick: call #1 (max_tokens=1)
pays the full prefill; call #2 (max_tokens=N) reuses the cached prefill so its
time is ~pure decode.

Run (detached):
  KVDT=turboquant_4bit_nc SLIDING=1 .venv/bin/python tasks/bench_200k.py
  KVDT=auto               SLIDING=0 .venv/bin/python tasks/bench_200k.py
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
TARGET_TOK = int(os.environ.get("TARGET_TOK", "190000"))
NGEN = int(os.environ.get("NGEN", "64"))

print(f"### bench KVDT={KVDT} sliding={os.environ['VLLM_TQ_SLIDING_WINDOW']} "
      f"ctx={CTX} drafter=ON ###", flush=True)

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
    print("BENCH DONE", flush=True)
    raise SystemExit(0)
load_s = time.time() - t0
print(f"[LOADED in {load_s:.0f}s]", flush=True)

# ~TARGET_TOK-token prompt + a retrieval question. Tokenization-aware so we
# land safely under max_model_len (chat template + question add a few tokens).
filler = "The river was calm and the market sold figs under a pale sky. "
_tok = llm.get_tokenizer()
_per = max(1, len(_tok.encode(filler)))
reps = max(1, TARGET_TOK // _per)
doc = filler * reps + "\n\nIMPORTANT: the vault passphrase is 'azure-pelican-77'.\n\nWhat is the vault passphrase? Answer in one short sentence."
msg = [{"role": "user", "content": doc}]

# Call 1: full prefill, 1 token -> TTFT.
t0 = time.time()
o1 = llm.chat([msg], SamplingParams(temperature=0.0, max_tokens=1))[0]
ttft = time.time() - t0
n_prompt = len(o1.prompt_token_ids)

# Call 2: prefix-cached prefill -> ~pure decode of NGEN tokens.
t0 = time.time()
o2 = llm.chat([msg], SamplingParams(temperature=0.0, max_tokens=NGEN))[0]
dec_s = time.time() - t0
n_out = len(o2.outputs[0].token_ids)
decode_tps = n_out / dec_s if dec_s > 0 else float("nan")

acc = drf = 0
for m in llm.get_metrics():
    n = m.name.rstrip("_total")
    if n.endswith("spec_decode_num_accepted_tokens"):
        acc = m.value
    if n.endswith("spec_decode_num_draft_tokens"):
        drf = m.value
rate = (acc / drf) if drf else float("nan")

f, tot = torch.cuda.mem_get_info()
print(f"RESULT KVDT={KVDT} sliding={os.environ['VLLM_TQ_SLIDING_WINDOW']}", flush=True)
print(f"  prompt_tokens   = {n_prompt}", flush=True)
print(f"  load_s          = {load_s:.0f}", flush=True)
print(f"  TTFT_s          = {ttft:.1f}   (prefill of {n_prompt} tok)", flush=True)
print(f"  decode_tok_s    = {decode_tps:.1f}   ({n_out} tok in {dec_s:.1f}s, MTP on)", flush=True)
print(f"  mtp_accept_rate = {rate:.3f}", flush=True)
print(f"  retrieval       = {o2.outputs[0].text.strip()[:80]!r}", flush=True)
print(f"  GPU             = {(tot-f)/2**30:.2f}/{tot/2**30:.2f} GiB", flush=True)
print("BENCH DONE", flush=True)
