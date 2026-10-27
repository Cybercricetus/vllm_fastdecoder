"""Online self-check of the FastDecode attention path (FASTDECODE_VERIFY=1).

Slow, for debugging only. For the selected layers (FASTDECODE_VERIFY_LAYERS:
comma-separated layer indices, default "0"; "all" for every layer) it

  * fingerprints every KV write in every step, using vLLM's own slot_mapping
    and positions: for each slot, the position written there and the sum of
    the written K row;
  * on every step that takes the split path, re-derives everything from vLLM's
    own GPU-side metadata (block_table, seq_lens, query_start_loc), not from
    FastDecode's host plan, and checks per request
      - the plan's host seq_lens equal vLLM's GPU seq_lens,
      - every position the request attends to holds the K that was last
        written for exactly that position (catches lost or misplaced writes,
        and blocks reused while still referenced),
      - the attention output rows match an fp32 reference computed from the
        same KV (read from whichever tier each block id maps to).

All-GPU requests in the same step ("P") are checked too; their error is the
stock-FlashAttention-vs-fp32 noise level to compare the others against.

Logs a summary every FASTDECODE_VERIFY_EVERY (default 20) checked steps and the
first FASTDECODE_VERIFY_MAX_DETAILS (default 30) individual violations.
"""

from __future__ import annotations

import logging
import os
import re

import torch

from fastdecode import state

logger = logging.getLogger("fastdecode")

_LAYER_IDX = re.compile(r"\.(\d+)\.")


def enabled() -> bool:
    return os.environ.get("FASTDECODE_VERIFY", "0").strip() not in ("", "0")


class Verifier:
    def __init__(self) -> None:
        spec = os.environ.get("FASTDECODE_VERIFY_LAYERS", "0").strip()
        self.all_layers = spec == "all"
        self.layers = set() if self.all_layers else {int(x) for x in spec.split(",") if x.strip()}
        self.every = int(os.environ.get("FASTDECODE_VERIFY_EVERY", "20"))
        self.tol = float(os.environ.get("FASTDECODE_VERIFY_TOL", "2e-2"))
        self.max_details = int(os.environ.get("FASTDECODE_VERIFY_MAX_DETAILS", "30"))
        self.fp: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.details = 0
        self.checked_steps = 0
        self.total_bad = 0
        self._reset()
        logger.warning(
            "FastDecode VERIFY on: layers=%s tol=%g (slow; diagnostics only)",
            "all" if self.all_layers else sorted(self.layers), self.tol,
        )

    def _reset(self) -> None:
        # class -> [n_requests, max_rel_err]
        self.acc = {k: [0, 0.0] for k in ("P", "S_cpu_only", "S_mixed", "F")}
        self.cnt = dict(seq_mismatch=0, pos_mismatch=0, unwritten=0, ksum_mismatch=0,
                        out_mismatch=0, nan=0)

    def selected(self, layer_name: str) -> bool:
        if self.all_layers:
            return True
        m = _LAYER_IDX.search(layer_name)
        return m is not None and int(m.group(1)) in self.layers

    # ------------------------------------------------------------------ writes

    def _tables(self, layer_name: str, device) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.fp.get(layer_name)
        if t is None:
            s = state.STORE
            n = (s.num_gpu_blocks + s.num_cpu_blocks) * s.block_size
            t = (torch.full((n,), -1, dtype=torch.int32, device=device),
                 torch.zeros(n, dtype=torch.float32, device=device))
            self.fp[layer_name] = t
        return t

    @staticmethod
    def _num_reqs(md) -> int:
        n = getattr(md, "fd_num_reqs", None)
        return int(n) if n is not None else int(md.query_start_loc.numel()) - 1

    @torch.no_grad()
    def record(self, layer_name: str, key: torch.Tensor, md) -> None:
        pos_t, sum_t = self._tables(layer_name, key.device)
        R = self._num_reqs(md)
        n_tok = int(md.num_actual_tokens)
        if R == 0 or n_tok == 0:
            return
        dev = key.device
        qsl = md.query_start_loc[: R + 1].long()
        seq = md.seq_lens[:R].long()
        qlen = qsl[1:] - qsl[:-1]
        ctx = seq - qlen
        req = torch.repeat_interleave(torch.arange(R, device=dev), qlen, output_size=n_tok)
        pos = ctx[req] + torch.arange(n_tok, device=dev) - qsl[:-1][req]
        slots = md.slot_mapping[:n_tok].long()
        ok = slots >= 0
        pos_t[slots[ok]] = pos[ok].to(torch.int32)
        sum_t[slots[ok]] = key[:n_tok][ok].float().sum(dim=(1, 2))

    # ------------------------------------------------------------------ check

    def _detail(self, msg: str, *args) -> None:
        self.total_bad += 1
        if self.details < self.max_details:
            self.details += 1
            logger.warning("FastDecode VERIFY violation: " + msg, *args)

    @torch.no_grad()
    def check(self, impl, layer, query, key, value, kv_cache, md, output) -> None:
        store = state.STORE
        name = layer.layer_name
        G, bs = store.num_gpu_blocks, store.block_size
        H, KVH, D = impl.num_heads, impl.num_kv_heads, impl.head_size
        grp = H // KVH
        dev = query.device
        pos_t, sum_t = self._tables(name, dev)
        R = self._num_reqs(md)
        qsl = md.query_start_loc[: R + 1].long().cpu()
        seq = md.seq_lens[:R].long().cpu()
        bt = md.block_table[:R].long().cpu()

        host_seq = getattr(md, "fd_seq_cpu", None)
        if host_seq is not None:
            bad = (host_seq[:R].long() != seq).nonzero().flatten()
            if bad.numel():
                self.cnt["seq_mismatch"] += int(bad.numel())
                i = int(bad[0])
                self._detail("%s: host seq_lens != GPU seq_lens for %d reqs (e.g. req %d: "
                             "host %d vs gpu %d)", name, bad.numel(), i,
                             int(host_seq[i]), int(seq[i]))

        kc, vc = kv_cache.transpose(1, 2).split(D, dim=-1)  # [B, bs, KVH, D]
        ck, cv = store.k[name], store.v[name]  # [C, KVH, bs, D]
        for i in range(R):
            r0, r1 = int(qsl[i]), int(qsl[i + 1])
            ql = r1 - r0
            if ql <= 0:
                continue
            L = int(seq[i])
            ctx = L - ql
            nb = (L + bs - 1) // bs
            ids = bt[i, :nb]
            on_cpu = ids >= G
            n_cpu = int(on_cpu.sum())
            K = torch.empty(nb, bs, KVH, D, dtype=kc.dtype, device=dev)
            V = torch.empty_like(K)
            gi = (~on_cpu).nonzero().flatten()
            ci = on_cpu.nonzero().flatten()
            if gi.numel():
                K[gi.to(dev)] = kc[ids[gi].to(dev)]
                V[gi.to(dev)] = vc[ids[gi].to(dev)]
            if ci.numel():
                K[ci.to(dev)] = ck[ids[ci] - G].transpose(1, 2).to(dev)
                V[ci.to(dev)] = cv[ids[ci] - G].transpose(1, 2).to(dev)
            K = K.reshape(nb * bs, KVH, D)[:L]
            V = V.reshape(nb * bs, KVH, D)[:L]

            if n_cpu == 0:
                cls = "P"
            elif ql > 1:
                cls = "F"
            else:
                cls = "S_mixed" if n_cpu < nb else "S_cpu_only"
            where = (f"{name} req {i} cls={cls} L={L} q={ql} blocks={nb} "
                     f"cpu_blocks={n_cpu} ids={ids.tolist()}")

            # 1) the KV each position holds is what was last written there
            p = torch.arange(L)
            slots = (ids.repeat_interleave(bs)[:L] * bs + p % bs).to(dev)
            fpos = pos_t[slots]
            unwritten = int((fpos == -1).sum())
            pos_bad = (fpos != p.to(dev)) & (fpos != -1)
            n_pos_bad = int(pos_bad.sum())
            ksum = K.float().sum(dim=(1, 2))
            fsum = sum_t[slots]
            sum_bad = ((ksum - fsum).abs() > 1e-3 * (1 + fsum.abs())) & (fpos != -1)
            n_sum_bad = int(sum_bad.sum())
            if unwritten:
                self.cnt["unwritten"] += unwritten
                first = int((fpos == -1).nonzero()[0])
                self._detail("%s: %d attended positions were never written (first pos %d)",
                             where, unwritten, first)
            if n_pos_bad:
                self.cnt["pos_mismatch"] += n_pos_bad
                first = int(pos_bad.nonzero()[0])
                self._detail("%s: %d positions hold KV written for another position "
                             "(first pos %d holds pos %d) -> block reused / misplaced write",
                             where, n_pos_bad, first, int(fpos[first]))
            if n_sum_bad:
                self.cnt["ksum_mismatch"] += n_sum_bad
                first = int(sum_bad.nonzero()[0])
                self._detail("%s: %d positions hold different K than was written (first "
                             "pos %d, block %d on %s)", where, n_sum_bad, first,
                             int(ids[first // bs]), "CPU" if bool(on_cpu[first // bs]) else "GPU")

            # 2) attention output vs fp32 reference (fresh k/v for this step's tokens)
            q = query[r0:r1].float()
            Kr = torch.cat([K[:ctx], key[r0:r1]]).float().repeat_interleave(grp, dim=1)
            Vr = torch.cat([V[:ctx], value[r0:r1]]).float().repeat_interleave(grp, dim=1)
            s = torch.einsum("qhd,khd->hqk", q, Kr) * impl.scale
            if ql > 1:
                qi = torch.arange(ql, device=dev)[:, None] + ctx
                kj = torch.arange(L, device=dev)[None, :]
                s = s.masked_fill((kj > qi)[None], float("-inf"))
            ref = torch.einsum("hqk,khd->qhd", torch.softmax(s, dim=-1), Vr)
            got = output[r0:r1].float()
            if not bool(torch.isfinite(got).all()):
                self.cnt["nan"] += 1
                self._detail("%s: output has NaN/Inf", where)
                err = float("inf")
            else:
                err = float((got - ref).abs().max() / (ref.abs().max() + 1e-3))
            a = self.acc[cls]
            a[0] += 1
            a[1] = max(a[1], err)
            if not err <= self.tol:
                self.cnt["out_mismatch"] += 1
                self._detail("%s: attention output rel err %.3g > %.3g", where, err, self.tol)

        self.checked_steps += 1
        if self.checked_steps % self.every == 0:
            self._summary(name)

    def _summary(self, name: str) -> None:
        cls = ", ".join(f"{k} n={v[0]} max_rel_err={v[1]:.2e}" for k, v in self.acc.items())
        cnt = ", ".join(f"{k}={v}" for k, v in self.cnt.items())
        logger.warning("FastDecode VERIFY summary (%d checked layer-steps, %d violations "
                       "so far; last %d): %s | %s", self.checked_steps, self.total_bad,
                       self.every, cls, cnt)
        self._reset()
