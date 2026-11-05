#!/usr/bin/env python
# TT side of the Qwen3-0.6B live-migration experiment: the same decoder
# layer math (layer 0, bf16, GQA decode with a growing KV cache) expressed
# in ttnn ops, run on a simulated Tenstorrent Blackhole.
#
#   qwen_tt.py ttfull  --steps N              -> TT-only full run (tolerance
#                                                reference for cross-backend)
#   qwen_tt.py target  --state qwen_state.npz -> restore the NV checkpoint
#                                                (bit-exact) and finish
#                                                steps K..N on TT
#
# Verification: the restored state must hash bit-identically to the NV
# capture; the resumed result is compared to the NV-only run under a bf16
# tolerance (different backends reduce in different orders).

import argparse
import hashlib
import json
import os

import numpy as np
import torch
import ttnn
from safetensors import safe_open

SEED = 0xC0FFEE
DTYPE = torch.bfloat16          # switched to float32 by --dtype fp32
TT_DTYPE = ttnn.bfloat16
SNAP = os.environ.get("QWEN_SNAP", os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"))
NH_Q, NH_KV, HD, HID, INTER = 16, 8, 128, 1024, 3072
ROPE_THETA = 1000000.0


def load_weights():
    if os.path.isdir(SNAP):
        f = sorted(glob.glob(os.path.join(SNAP, "*.safetensors")))[0]
    else:
        f = SNAP
    w = {}
    with safe_open(f, framework="pt") as sf:
        for k in sf.keys():
            if k.startswith("model.layers.0."):
                w[k.split("model.layers.0.")[1]] = sf.get_tensor(k).float()
    # pack qkv for one linear; ttnn.linear wants [in, out] weights
    w["qkv"] = torch.cat([w["self_attn.q_proj.weight"],
                          w["self_attn.k_proj.weight"],
                          w["self_attn.v_proj.weight"]], dim=0).t().contiguous().to(DTYPE)
    for k in list(w):
        if k != "qkv":
            w[k] = w[k].t().contiguous().to(torch.bfloat16)
    return w


def rope_cs(pos, device_cpu=True):
    """Cos/sin tables for one position, fp32, matching HF's rotary exactly."""
    inv = 1.0 / (ROPE_THETA ** (torch.arange(0, HD, 2, dtype=torch.float32) / HD))
    fr = pos * inv
    emb = torch.cat([fr, fr])
    return emb.cos(), emb.sin()


def to_tt(dev, t):
    return ttnn.from_torch(t.contiguous(), dtype=TT_DTYPE,
                           layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def from_tt(t):
    return ttnn.to_torch(t).float()


def fp32_rms_norm(dev, x_tt, w_tt_bf, eps=1e-6):
    """RMSNorm in fp32, matching HF's Qwen3RMSNorm: fp32 arithmetic,
    bf16 weight, bf16 output. ttnn.rms_norm computes in bf16 on this
    target and drifts ~6% from the reference formula."""
    x = ttnn.typecast(x_tt, ttnn.float32)
    ms = ttnn.mean(ttnn.multiply(x, x, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                   dim=-1, keepdim=True, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    inv = ttnn.add(ms, eps, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    inv = ttnn.rsqrt(inv, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    w32 = ttnn.typecast(w_tt_bf, ttnn.float32)
    y = ttnn.multiply(ttnn.multiply(x, inv, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                      w32, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.typecast(y, TT_DTYPE)


def step_tt(dev, W, h_tt, cos_q, sin_q, cos_k, sin_k, k_host, v_host, step_idx):
    """One decode step. k_host/v_host: python lists of [8,1,128] float32
    arrays (the logical KV cache; also what the state file carries)."""
    # input layernorm
    ln = fp32_rms_norm(dev, h_tt, W["input_layernorm.weight"])
    # qkv projection: [1,1024] -> [1,4096]
    qkv = ttnn.linear(ln, W["qkv_tt"], memory_config=ttnn.DRAM_MEMORY_CONFIG)
    qkv = ttnn.reshape(qkv, ttnn.Shape([1, (NH_Q + 2 * NH_KV) * HD]))
    q = ttnn.slice(qkv, ttnn.Shape([0, 0]), ttnn.Shape([1, NH_Q * HD]),
                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k = ttnn.slice(qkv, ttnn.Shape([0, NH_Q * HD]),
                   ttnn.Shape([1, (NH_Q + NH_KV) * HD]),
                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    v = ttnn.slice(qkv, ttnn.Shape([0, (NH_Q + NH_KV) * HD]),
                   ttnn.Shape([1, (NH_Q + 2 * NH_KV) * HD]),
                   memory_config=ttnn.DRAM_MEMORY_CONFIG)
    q = ttnn.reshape(q, ttnn.Shape([NH_Q, HD]))
    k = ttnn.reshape(k, ttnn.Shape([NH_KV, HD]))
    v = ttnn.reshape(v, ttnn.Shape([NH_KV, HD]))
    # per-head qk-norm (rmsnorm over head_dim)
    q = fp32_rms_norm(dev, q, W["self_attn.q_norm.weight"])
    k = fp32_rms_norm(dev, k, W["self_attn.k_norm.weight"])
    # rope: q = q*cos + rot_half(q)*sin, tables pre-expanded per head
    qh = ttnn.reshape(q, ttnn.Shape([NH_Q, 1, HD]))
    kh = ttnn.reshape(k, ttnn.Shape([NH_KV, 1, HD]))
    x1 = ttnn.slice(qh, ttnn.Shape([0, 0, 0]), ttnn.Shape([NH_Q, 1, HD // 2]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x2 = ttnn.slice(qh, ttnn.Shape([0, 0, HD // 2]), ttnn.Shape([NH_Q, 1, HD]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rot = ttnn.concat([neg(dev, x2), x1], dim=-1,
                      memory_config=ttnn.DRAM_MEMORY_CONFIG)
    qe = ttnn.add(
        ttnn.multiply(qh, cos_q, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.multiply(rot, sin_q, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k1 = ttnn.slice(kh, ttnn.Shape([0, 0, 0]), ttnn.Shape([NH_KV, 1, HD // 2]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    k2 = ttnn.slice(kh, ttnn.Shape([0, 0, HD // 2]), ttnn.Shape([NH_KV, 1, HD]),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    rotk = ttnn.concat([neg(dev, k2), k1], dim=-1,
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ke = ttnn.add(
        ttnn.multiply(kh, cos_k, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        ttnn.multiply(rotk, sin_k, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    # cache assembly on host: ke/v come back as [8,1,128]; GQA expand on
    # the way in (each KV head feeds the 2 adjacent Q heads)
    ke_h = from_tt(ke).reshape(NH_KV, 1, HD)
    v_h = from_tt(v).reshape(NH_KV, 1, HD)
    k_host.append(ke_h)
    v_host.append(v_h)
    S = len(k_host)
    K_all = np.concatenate(k_host, axis=1)           # [8,S,128]
    V_all = np.concatenate(v_host, axis=1)
    K_pair = np.concatenate([K_all, K_all], axis=1).reshape(NH_KV * 2, S, HD)
    V_pair = np.concatenate([V_all, V_all], axis=1).reshape(NH_KV * 2, S, HD)
    K_t = to_tt(dev, torch.from_numpy(np.ascontiguousarray(
        np.transpose(K_pair, (0, 2, 1)))))           # [16,128,S]
    V_t = to_tt(dev, torch.from_numpy(np.ascontiguousarray(V_pair)))
    # scores: q [16,1,128] x K_pair [16,S,128] -> [16,1,S]
    scale = HD ** -0.5
    scores = ttnn.matmul(qe, K_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    scores = ttnn.multiply(scores, scale, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    probs = ttnn.softmax(scores, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn = ttnn.matmul(probs, V_t, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    attn = ttnn.reshape(attn, ttnn.Shape([1, NH_Q * HD]))
    o = ttnn.linear(attn, W["self_attn.o_proj.weight"],
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
    h1 = ttnn.add(h_tt, o, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ln2 = fp32_rms_norm(dev, h1, W["post_attention_layernorm.weight"])
    gate = ttnn.linear(ln2, W["mlp.gate_proj.weight"],
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    up = ttnn.linear(ln2, W["mlp.up_proj.weight"],
                     memory_config=ttnn.DRAM_MEMORY_CONFIG)
    mid = ttnn.multiply(ttnn.silu(gate, memory_config=ttnn.DRAM_MEMORY_CONFIG),
                        up, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    down = ttnn.linear(mid, W["mlp.down_proj.weight"],
                       memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.add(h1, down, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def neg(dev, x):
    return ttnn.multiply(x, -1.0, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def vh(dev, v):
    return ttnn.reshape(v, ttnn.Shape([NH_KV, 1, HD]))


def rope_tables(step):
    cos, sin = rope_cs(step)
    # expand: q [16,1,128] tables, k [8,1,128]
    cq = cos.reshape(1, 1, HD).expand(NH_Q, 1, HD).to(DTYPE)
    sq = sin.reshape(1, 1, HD).expand(NH_Q, 1, HD).to(DTYPE)
    ck = cos.reshape(1, 1, HD).expand(NH_KV, 1, HD).to(DTYPE)
    sk = sin.reshape(1, 1, HD).expand(NH_KV, 1, HD).to(DTYPE)
    return cq, sq, ck, sk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ttfull", "target", "model_target"])
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--state", default="results/qwen_state.npz")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    args = ap.parse_args()
    global DTYPE, TT_DTYPE
    if args.dtype == "fp32":
        DTYPE, TT_DTYPE = torch.float32, ttnn.float32

    W = load_weights()
    dev = ttnn.CreateDevice(device_id=0)
    Wt = {k: to_tt(dev, v) for k, v in W.items()}
    Wt["qkv_tt"] = Wt.pop("qkv")

    if args.mode == "model_target":
        run_model_target(dev, args)
        ttnn.CloseDevice(dev)
        return

    if args.mode == "ttfull":
        g = torch.Generator().manual_seed(SEED)
        h = torch.randn(1, 1, HID, generator=g).to(torch.bfloat16)
        ks, vs = [], []
        h_tt = to_tt(dev, h)
        for s in range(args.steps):
            cq, sq, ck, sk = rope_tables(s)
            h_tt = step_tt(dev, Wt, h_tt, to_tt(dev, cq), to_tt(dev, sq),
                           to_tt(dev, ck), to_tt(dev, sk), ks, vs, s)
        hf = from_tt(h_tt).numpy()
        np.savez("results/qwen_ttfull.npz", final=hf, steps=np.array(args.steps))
        ref = np.load("results/qwen_full.npz")["final"]
        d = np.abs(hf - ref)
        print(f"TT-full vs NV-full: max abs diff {d.max():.4f}, "
              f"rel {(d.max() / np.abs(ref).max()):.4f}")
        ttnn.CloseDevice(dev)
        return

    # target: restore NV checkpoint at step K, finish on TT
    st = np.load(args.state)
    ann = json.load(open(args.state + ".ctx.json"))
    h0 = torch.from_numpy(st["hidden_u16"]).view(torch.bfloat16)
    k0 = torch.from_numpy(st["keys_u16"]).view(torch.bfloat16)   # [1,8,S,128]
    v0 = torch.from_numpy(st["values_u16"]).view(torch.bfloat16)
    S = k0.shape[-2]
    step0 = int(st["step"])

    # bit-exactness of the restored state itself
    hh = hashlib.sha256(h0.contiguous().view(torch.uint16).numpy().tobytes()
                        + k0.contiguous().view(torch.uint16).numpy().tobytes()
                        + v0.contiguous().view(torch.uint16).numpy().tobytes()).hexdigest()[:16]
    bit_exact = (hh == st["state_sha"])
    print(f"restored state sha {hh} vs captured {st['state_sha']} -> "
          f"bit_exact={bit_exact}")

    h_tt = to_tt(dev, h0.reshape(1, 1, HID))
    k_host, v_host = [], []
    kvk = k0[0].contiguous()   # [8, S, 128] bf16
    kvv = v0[0].contiguous()
    for i in range(S):
        k_host.append(kvk[:, i:i + 1, :].float().numpy())
        v_host.append(kvv[:, i:i + 1, :].float().numpy())

    for s in range(step0, args.steps):
        cq, sq, ck, sk = rope_tables(s)
        h_tt = step_tt(dev, Wt, h_tt, to_tt(dev, cq), to_tt(dev, sq),
                       to_tt(dev, ck), to_tt(dev, sk), k_host, v_host, s)
    hf = from_tt(h_tt).numpy()
    ref = np.load("results/qwen_full.npz")["final"]
    d = np.abs(hf - ref)
    cos = float(np.dot(hf.reshape(-1), ref.reshape(-1)) /
                (np.linalg.norm(hf) * np.linalg.norm(ref) + 1e-9))
    res = {"bit_exact_state": bool(bit_exact), "max_abs_diff": float(d.max()),
           "rel_diff": float(d.max() / np.abs(ref).max()), "cosine": cos,
           "nv_steps": args.steps, "k": step0}
    json.dump(res, open("results/qwen_nvtt_result.json", "w"), indent=1)
    print(f"resumed on TT: steps {step0}..{args.steps - 1} | "
          f"max abs diff {d.max():.4f} | rel {res['rel_diff']:.4f} | "
          f"cos {cos:.6f}")
    ttnn.CloseDevice(dev)


def load_model_weights():
    """All 28 layers as per-layer weight dicts (ttnn.linear wants [in,out]),
    plus the final norm weight, the embedding table and lm_head kept on the
    host (token selection is runtime logic)."""
    f = SNAP if SNAP.endswith(".safetensors") else \
        sorted(glob.glob(os.path.join(SNAP, "*.safetensors")))[0]
    layers, embed, lm_head, final_norm = [], None, None, None
    with safe_open(f, framework="pt") as sf:
        emb_w = sf.get_tensor("model.embed_tokens.weight").float()
        lm = sf.get_tensor("lm_head.weight").float()
        final_norm = sf.get_tensor("model.norm.weight").to(DTYPE)
        n_layers = 28
        per = [dict() for _ in range(n_layers)]
        for k in sf.keys():
            if ".layers." not in k:
                continue
            parts = k.split(".")
            li = int(parts[2])
            name = ".".join(parts[3:])
            t = sf.get_tensor(k)
            t = t.t().contiguous() if t.dim() == 2 else t
            per[li][name] = t.to(torch.bfloat16)
        for li in range(n_layers):
            d = per[li]
            d["qkv"] = torch.cat([d.pop("self_attn.q_proj.weight"),
                                  d.pop("self_attn.k_proj.weight"),
                                  d.pop("self_attn.v_proj.weight")], dim=1)
            layers.append(d)
        embed = emb_w
        lm_head = lm
    return layers, embed, lm_head, final_norm


def run_model_target(dev, args):
    st = np.load(args.state)
    ann = json.load(open(args.state + ".ctx.json"))
    hh = hashlib.sha256(st["keys_u16"].tobytes() + st["values_u16"].tobytes()
                        + np.array(int(st["last_token"]),
                                   dtype=np.int64).tobytes()).hexdigest()[:16]
    bit_exact = (hh == st["state_sha"])
    print(f"restored 28-layer state sha {hh} vs captured {st['state_sha']} "
          f"-> bit_exact={bit_exact}")

    layers_w, embed, lm_head, final_norm = load_model_weights()
    print("weights loaded:", len(layers_w), "layers")

    # host-side KV caches: per layer, list of [8,1,128] fp32 arrays
    k_host = [[None] * 0 for _ in layers_w]
    v_host = [[None] * 0 for _ in layers_w]
    ku = torch.from_numpy(st["keys_u16"]).view(torch.bfloat16)   # [L,1,8,S,128]
    vu = torch.from_numpy(st["values_u16"]).view(torch.bfloat16)
    S = ku.shape[-2]
    for li in range(len(layers_w)):
        kl = ku[li, 0].contiguous()
        vl = vu[li, 0].contiguous()
        for i in range(S):
            k_host[li].append(kl[:, i:i + 1, :].float().numpy())
            v_host[li].append(vl[:, i:i + 1, :].float().numpy())

    cur = int(st["last_token"])
    toks = [cur]
    h = None
    step0 = int(st["step"])
    Wt_layers = [{k: to_tt(dev, v) for k, v in L.items()} for L in layers_w]
    for Lw in Wt_layers:
        Lw["qkv_tt"] = Lw.pop("qkv")
    print("weights uploaded to device")
    for s in range(step0, args.steps):
        h = embed[torch.tensor(cur)].to(DTYPE).reshape(1, 1, HID)
        h_tt = to_tt(dev, h)
        for li in range(len(layers_w)):
            h_tt = step_tt(dev, Wt_layers[li], h_tt,
                           *_rope_for(dev, s), k_host[li], v_host[li], s)
        # final norm (fp32) + host lm_head argmax
        ln = fp32_rms_norm(dev, h_tt, to_tt(dev, final_norm.reshape(1, HID)))
        hn = from_tt(ln).reshape(HID)
        logits = lm_head @ hn
        cur = int(np.argmax(logits))
        toks.append(cur)
        print(f"  step {s}: token {cur}")

    ref = np.load("results/qwen_model_full.npz")
    ref_toks = ref["tokens"].tolist()
    match = sum(1 for a, b in zip(toks, ref_toks) if a == b)
    hd = np.abs(hf_last if False else from_tt(h_tt).numpy() - ref["final_hidden"])
    res = {"bit_exact_state": bool(bit_exact), "tokens": toks,
           "ref_tokens": ref_toks, "token_match": f"{match}/{len(ref_toks)}",
           "final_hidden_max_diff": float(hd.max())}
    json.dump(res, open("results/qwen_model_nvtt_result.json", "w"), indent=1)
    print(f"model migration: tokens TT {toks}")
    print(f"                 NV  {ref_toks}")
    print(f"token match {match}/{len(ref_toks)}; "
          f"final hidden max diff {hd.max():.4f}")


def _rope_for(dev, s):
    cq, sq, ck, sk = rope_tables(s)
    return to_tt(dev, cq), to_tt(dev, sq), to_tt(dev, ck), to_tt(dev, sk)


if __name__ == "__main__":
    main()
