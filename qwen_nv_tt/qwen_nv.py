#!/usr/bin/env python
# NV side of the Qwen3-0.6B live-migration experiment: run decoder layer 0
# autoregressively on the real GPU (RTX PRO 6000, bf16, eager attention),
# checkpointing the layer's execution state at token-step boundaries.
#
#   qwen_nv.py full            --steps N            -> final hidden + logits-ish norm
#   qwen_nv.py source          --steps N --k K      -> state file at step K
#   qwen_nv.py harvest         --steps 1            -> torch.compile the step
#                                                     fn, copy inductor PTX out
#
# State (device-neutral, CTX-annotated): hidden states [1,1,1024] bf16,
# layer-0 KV cache (K,V [1,8,S,128] bf16), step counter, and the DWARF-
# recovered CTX annotation of the captured kernels.

import argparse
import glob
import hashlib
import json
import os
import shutil

import numpy as np
import torch
from transformers import AutoModelForCausalLM

TORCH_DTYPE = torch.bfloat16
from transformers.cache_utils import DynamicCache

MODEL = "Qwen/Qwen3-0.6B"
LAYER = 0
HID = 1024


def load_layer():
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=getattr(__import__("__main__"), "TORCH_DTYPE",
                                   torch.bfloat16),
        attn_implementation="eager", device_map="cuda")
    m.eval()
    return m.model.layers[LAYER], m.model.rotary_emb


def seed_hidden(step=0):
    # CPU generator: the same seed must produce the same input on every
    # backend (CUDA and CPU RNG streams differ)
    g = torch.Generator().manual_seed(0xC0FFEE)
    return torch.randn(1, 1, HID, generator=g).to(torch.bfloat16).to("cuda")


def step(layer, rot, hidden, cache, pos):
    pos_ids = torch.tensor([[pos]], device="cuda")
    cos, sin = rot(hidden.float(), pos_ids)
    out = layer(hidden,
                position_ids=pos_ids,
                past_key_values=cache,
                use_cache=True,
                position_embeddings=(cos.to(hidden.dtype), sin.to(hidden.dtype)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["full", "source", "harvest",
                                     "model_full", "model_source"])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default="results/qwen_state.npz")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    global TORCH_DTYPE
    TORCH_DTYPE = torch.float32 if args.dtype == "fp32" else torch.bfloat16
    os.makedirs("results", exist_ok=True)
    torch.manual_seed(0xC0FFEE)

    if args.mode in ("model_full", "model_source"):
        run_model(args)
        return

    layer, rot = load_layer()

    if args.mode == "harvest":
        cache = DynamicCache()
        h = seed_hidden()

        def compiled_step(h, pos):
            c = DynamicCache()
            c.update(h.new_zeros(1, 8, pos, 128) if pos else
                     torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda"),
                     torch.zeros(1, 8, 1, 128, dtype=torch.bfloat16, device="cuda"),
                     0)
            return step(layer, rot, h, c, pos)

        cs = torch.compile(compiled_step, dynamic=False)
        for p in range(2):
            cs(h, p)
        torch.cuda.synchronize()
        cache_dir = "/tmp/inductor_ptx"
        shutil.rmtree(cache_dir, ignore_errors=True)
        os.makedirs(cache_dir, exist_ok=True)
        n = 0
        for f in glob.glob(os.path.expanduser("~/.cache/torch_inductor/**/*ptx*"),
                           recursive=True):
            if f.endswith(".ptx"):
                shutil.copy(f, f"{cache_dir}/kernel_{n:02d}.ptx")
                n += 1
        print(f"harvested {n} inductor PTX kernels -> {cache_dir}")
        return

    # run N steps from a fresh seed chain
    cache = DynamicCache()
    h = seed_hidden()
    for s in range(args.steps):
        h = step(layer, rot, h, cache, s)
    final = h.detach().float().cpu().numpy()

    if args.mode == "full":
        np.savez("results/qwen_full.npz", final=final,
                 steps=np.array(args.steps))
        print(f"NV full: {args.steps} steps, final |h| = "
              f"{np.abs(final).mean():.4f}")
        return

    # source: rerun deterministically and capture at step K
    cache = DynamicCache()
    h = seed_hidden()
    for s in range(args.k):
        h = step(layer, rot, h, cache, s)
    keys_u16 = cache.layers[0].keys.detach().cpu().contiguous().view(torch.uint16).numpy()   # [1,8,S,128] bit-exact
    values_u16 = cache.layers[0].values.detach().cpu().contiguous().view(torch.uint16).numpy()
    hnp = h.detach().cpu().contiguous().view(torch.uint16).numpy()
    state_hash = hashlib.sha256(
        hnp.tobytes() + keys_u16.tobytes() + values_u16.tobytes()).hexdigest()[:16]
    np.savez(args.out, hidden_u16=hnp, keys_u16=keys_u16, values_u16=values_u16,
             step=np.array(args.k), state_sha=state_hash)
    ann = {
        "format": "device-neutral/ctx-state-v1",
        "source_device": "nvidia-rtx-pro-6000-blackwell (sm_120, CUDA 13.2)",
        "workload": "Qwen3-0.6B decoder layer 0, bf16, eager attention",
        "safe_point": f"token-step boundary after step {args.k}",
        "live_values": ["hidden[1,1,1024]", "K_cache", "V_cache", "step"],
        "kernels_captured": "torch-inductor PTX, DWARF-recovered (see ctx_annotation.json)",
    }
    json.dump(ann, open(args.out + ".ctx.json", "w"), indent=1)
    print(f"NV source: {args.k}/{args.steps} steps, state sha {state_hash}, "
          f"{os.path.getsize(args.out)} B")


def run_model(args):
    """Full-model greedy decode: the whole Qwen3-0.6B (28 layers), a real
    autoregressive token chain. `model_full` = uninterrupted reference;
    `model_source` = checkpoint at step K (all-layer KV cache, last token,
    step counter)."""
    from transformers.cache_utils import DynamicCache
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=TORCH_DTYPE, attn_implementation="eager",
        device_map="cuda")
    m.eval()
    tok = torch.tensor([[9707]])   # fixed prompt token
    n_layers = m.config.num_hidden_layers

    def step_model(cache, cur, pos):
        out = m(cur, position_ids=torch.tensor([[pos]], device="cuda"),
                past_key_values=cache, use_cache=True, output_hidden_states=True)
        return out.logits[0, -1].argmax().view(1, 1), out.hidden_states[-1]

    if args.mode == "model_full":
        cache = DynamicCache()
        cur, toks, hlast = tok.cuda(), [], None
        for s in range(args.steps):
            cur, hlast = step_model(cache, cur, s)
            toks.append(int(cur))
        hf = hlast.float().detach().cpu().numpy()
        np.savez("results/qwen_model_full.npz",
                 tokens=np.array(toks), final_hidden=hf,
                 steps=np.array(args.steps))
        print(f"NV model full: {args.steps} steps, tokens {toks}")
        return

    cache = DynamicCache()
    cur, toks = tok.cuda(), []
    for s in range(args.steps):
        cur, _ = step_model(cache, cur, s)
        toks.append(int(cur))
    keys_u16 = torch.stack([c.keys.contiguous().view(torch.uint16)
                            for c in cache.layers]).cpu().numpy()   # [L,1,8,S,128]
    values_u16 = torch.stack([c.values.contiguous().view(torch.uint16)
                              for c in cache.layers]).cpu().numpy()
    state_hash = hashlib.sha256(keys_u16.tobytes() + values_u16.tobytes()
                                + np.array(toks[-1], dtype=np.int64).tobytes()
                                ).hexdigest()[:16]
    np.savez(args.out, keys_u16=keys_u16, values_u16=values_u16,
             last_token=np.array(toks[-1]), tokens=np.array(toks),
             step=np.array(args.k), state_sha=state_hash)
    ann = {
        "format": "device-neutral/ctx-state-v1",
        "source_device": "nvidia-rtx-pro-6000-blackwell (sm_120, CUDA 13.2)",
        "workload": f"Qwen3-0.6B full-model greedy decode ({n_layers} layers)",
        "safe_point": f"token-step boundary after step {args.k}",
        "live_values": ["KV_cache x28 layers", "last_token", "step"],
        "state_sha": state_hash,
    }
    json.dump(ann, open(args.out + ".ctx.json", "w"), indent=1)
    print(f"NV model source: {args.k}/{args.steps} steps, tokens {toks}, "
          f"sha {state_hash}")


if __name__ == "__main__":
    main()
