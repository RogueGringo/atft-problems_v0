#!/usr/bin/env python3
"""
CARET Figure 14.11 — CORRECTED transcoding.

Errors in v1:
  - Rings read as depth → they're CHANNELS
  - Line thickness read as bandwidth → it's WEIGHT CLASS {0,1,3}
  - Child rings read as sequential layers → they're internal channels

Corrected architecture:
  - Parent: 1 transformer block with 4-channel harmonic input (4 rings = 4 channels)
  - Connections: {0,1,3} weighted (crystal-proportional routing, not asymmetric)
  - Children: 1 layer each with 3 internal channels (3 rings = 3 channels)
  - The diagram IS the Harmonic Stack with harmonic transducer
"""
import json, time, sys, numpy as np, torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topological-router"))

from ternary_linear import TernaryLinear, make_linear
from ternary_transformer import GPTConfig, Block
from run_long import TextChunked, split_dataset, measure_perplexity, weight_stats
from topo_measures import effective_rank, spectral_gap, gini_fast


class CARETv2(nn.Module):
    """Corrected CARET transcoding: rings=channels, lines=weights."""

    def __init__(self, vocab_size=50257, n_embd=2048, n_head=16, block_size=512):
        super().__init__()
        self.block_size = block_size

        # === PARENT NODE: 4-ring = 4-channel harmonic input ===
        # Each ring is a channel embedding, not a depth layer
        # ch0: primary character embedding (largest ring = most information)
        # ch1-ch3: structural harmonic channels (smaller rings)
        ch0_dim = 1024  # outermost ring — widest, carries most
        ch1_dim = 384   # second ring
        ch2_dim = 384   # third ring
        ch3_dim = 256   # innermost ring — narrowest, most concentrated
        # Total: 2048

        self.ch0_emb = nn.Embedding(vocab_size, ch0_dim)
        self.ch1_emb = nn.Embedding(vocab_size, ch1_dim)
        self.ch2_emb = nn.Embedding(vocab_size, ch2_dim)
        self.ch3_emb = nn.Embedding(vocab_size, ch3_dim)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(0.0)

        # 4-channel lens: combines channels into unified representation
        # This IS the concentric ring integration — 4 rings → 1 signal
        self.lens = make_linear(n_embd, n_embd, bias=False, weight_set='013')

        # 1 transformer block (NOT 4 — rings are channels, not depth)
        gpt_cfg = GPTConfig(
            vocab_size=vocab_size, block_size=block_size,
            n_layer=1, n_head=n_head, n_embd=n_embd,
            dropout=0.0, weight_set='013'
        )
        self.prism = Block(gpt_cfg)
        self.ln_prism = nn.LayerNorm(n_embd)

        # === ROUTING: crystal-proportional, not asymmetric ===
        # Line thickness = weight class, not bandwidth
        # So routing uses the crystal ratio: 22/42/36
        n_void = int(n_embd * 0.222)    # 455
        n_ident = int(n_embd * 0.417)   # 854
        n_prime = n_embd - n_void - n_ident  # 739
        self.split_sizes = [n_void, n_ident, n_prime]

        # === CHILDREN: 1 layer each with 3 internal channels ===
        # 3 rings per child = 3-channel internal structure
        # NOT 3 sequential layers
        # Each channel is a separate {0,1,3} linear, outputs concatenated

        # Void analyzer: 3 internal channels
        v_ch = n_void // 3
        v_rem = n_void - 3 * v_ch
        self.void_ch0 = make_linear(n_void, v_ch + v_rem, bias=False, weight_set='013')
        self.void_ch1 = make_linear(n_void, v_ch, bias=False, weight_set='013')
        self.void_ch2 = make_linear(n_void, v_ch, bias=False, weight_set='013')
        self.void_ln = nn.LayerNorm(n_void)

        # Identity analyzer: 3 internal channels
        i_ch = n_ident // 3
        i_rem = n_ident - 3 * i_ch
        self.ident_ch0 = make_linear(n_ident, i_ch + i_rem, bias=False, weight_set='013')
        self.ident_ch1 = make_linear(n_ident, i_ch, bias=False, weight_set='013')
        self.ident_ch2 = make_linear(n_ident, i_ch, bias=False, weight_set='013')
        self.ident_ln = nn.LayerNorm(n_ident)

        # Prime analyzer: 3 internal channels
        p_ch = n_prime // 3
        p_rem = n_prime - 3 * p_ch
        self.prime_ch0 = make_linear(n_prime, p_ch + p_rem, bias=False, weight_set='013')
        self.prime_ch1 = make_linear(n_prime, p_ch, bias=False, weight_set='013')
        self.prime_ch2 = make_linear(n_prime, p_ch, bias=False, weight_set='013')
        self.prime_ln = nn.LayerNorm(n_prime)

        self.act = nn.GELU()

        # LM head (untied — recombined output is full n_embd width)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, return_hidden=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)

        # 4-channel embedding (4 concentric rings of the parent node)
        x = torch.cat([
            self.ch0_emb(idx),    # outermost ring — primary signal
            self.ch1_emb(idx),    # second ring — harmonic 1
            self.ch2_emb(idx),    # third ring — harmonic 2
            self.ch3_emb(idx),    # innermost ring — harmonic 3
        ], dim=-1)  # (B, T, 2048)

        # Lens layer: integrate 4 channels (the ring-to-signal transform)
        x = self.drop(self.act(self.lens(x)) + self.pos_emb(pos))

        # 1 prism block (single processing layer, not 4)
        x = self.prism(x)
        hidden = [x.detach()] if return_hidden else []
        x = self.ln_prism(x)

        # Crystal-proportional routing (line thickness = weight class)
        void_x, ident_x, prime_x = torch.split(x, self.split_sizes, dim=-1)

        # 3-channel children (rings = parallel channels, not sequential layers)
        void_out = self.void_ln(self.act(torch.cat([
            self.void_ch0(void_x), self.void_ch1(void_x), self.void_ch2(void_x)
        ], dim=-1)))

        ident_out = self.ident_ln(self.act(torch.cat([
            self.ident_ch0(ident_x), self.ident_ch1(ident_x), self.ident_ch2(ident_x)
        ], dim=-1)))

        prime_out = self.prime_ln(self.act(torch.cat([
            self.prime_ch0(prime_x), self.prime_ch1(prime_x), self.prime_ch2(prime_x)
        ], dim=-1)))

        # Recombine
        x = torch.cat([void_out, ident_out, prime_out], dim=-1)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1), ignore_index=-100)

        return (logits, loss, hidden) if return_hidden else (logits, loss)


def child_crystals(model):
    results = {}
    for name, layers in [
        ('void', [model.void_ch0, model.void_ch1, model.void_ch2]),
        ('identity', [model.ident_ch0, model.ident_ch1, model.ident_ch2]),
        ('prime', [model.prime_ch0, model.prime_ch1, model.prime_ch2]),
    ]:
        w0, w1, w3, total = 0, 0, 0, 0
        for layer in layers:
            if isinstance(layer, TernaryLinear):
                wq = layer.get_quantized_weight()
                n = wq.numel()
                w0 += (wq == 0).sum().item()
                w1 += (wq == 1).sum().item()
                w3 += (wq == 3).sum().item()
                total += n
        if total > 0:
            results[name] = {'w0': w0/total, 'w1': w1/total, 'w3': w3/total}
    return results


# === MAIN ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

ds = TextChunked(tokenizer, max_length=256, n_samples=200000, dataset_name="wikitext")
train_ds, test_ds = split_dataset(ds)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True,
                                            num_workers=2, pin_memory=True, drop_last=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False,
                                           num_workers=0, pin_memory=True, drop_last=True)
print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

model = CARETv2(vocab_size=tokenizer.vocab_size).to(device)
params = sum(p.numel() for p in model.parameters())
ternary = sum(p.numel() for n, p in model.named_parameters()
              if not any(k in n for k in ['emb', 'ln', 'lm_head']))
print(f"{params:,} params ({ternary/params*100:.1f}% ternary)")
print(f"Routing: void={model.split_sizes[0]} identity={model.split_sizes[1]} prime={model.split_sizes[2]}")

ternary_p = [p for n, p in model.named_parameters()
             if not any(k in n for k in ['emb', 'ln', 'lm_head'])]
continuous_p = [p for n, p in model.named_parameters()
                if any(k in n for k in ['emb', 'ln', 'lm_head'])]

optimizer = torch.optim.AdamW([
    {'params': ternary_p, 'weight_decay': 0.0},
    {'params': continuous_p, 'weight_decay': 0.01}
], lr=3e-4)

warmup = 2000; max_steps = 20000; accum = 4
def lr_lambda(step):
    if step < warmup: return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.05 + 0.95 * (1 + np.cos(np.pi * progress)) / 2
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

log = {"experiment": "CARET_v2_corrected", "checkpoints": [], "steps": []}
step = 0; start = time.time()
print(f"\nCARET v2 CORRECTED — rings=channels, lines=weights, 20K steps, L2=0.0\n")
model.train(); optimizer.zero_grad()

while step < max_steps:
    for bx, by in train_loader:
        if step >= max_steps: break
        bx, by = bx.to(device), by.to(device)
        _, loss = model(bx, targets=by)
        (loss / accum).backward()
        if (step + 1) % accum == 0:
            torch.nn.utils.clip_grad_norm_(continuous_p, 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
        if step % 200 == 0:
            ppl = torch.exp(loss).item()
            log["steps"].append({"step": step, "loss": loss.item(), "ppl": min(ppl, 1e7)})
            print(f"  {step:6d} | loss {loss.item():.4f} | ppl {ppl:.1f} | {time.time()-start:.0f}s", flush=True)
        if step > 0 and step % 1000 == 0:
            train_ppl = measure_perplexity(model, train_loader, device)
            test_ppl = measure_perplexity(model, test_loader, device)
            wstats = weight_stats(model)
            children = child_crystals(model)
            gap = test_ppl['ppl'] / train_ppl['ppl']
            ckpt = {'step': step, 'train_ppl': train_ppl, 'test_ppl': test_ppl,
                    'gen_gap': gap, 'weight_dist': wstats, 'child_crystals': children}
            log['checkpoints'].append(ckpt)
            print(f"\n    CKPT {step:6d} | train {train_ppl['ppl']:.1f} | test {test_ppl['ppl']:.1f} | gap {gap:.3f}")
            print(f"    crystal: 0={wstats['zero']:.3f} 1={wstats['one']:.3f} 3={wstats['three']:.3f}")
            for band, c in children.items():
                print(f"    child[{band:>8s}]: 0={c['w0']:.3f} 1={c['w1']:.3f} 3={c['w3']:.3f}")
            print(flush=True)
            with open('results/caret_v2_corrected/training_log.json', 'w') as f:
                json.dump(log, f, indent=2, default=str)
        step += 1

# Final
train_ppl = measure_perplexity(model, train_loader, device)
test_ppl = measure_perplexity(model, test_loader, device)
wstats = weight_stats(model)
children = child_crystals(model)
log['final'] = {'train_ppl': train_ppl, 'test_ppl': test_ppl,
                'weight_dist': wstats, 'child_crystals': children}

out = Path('results/caret_v2_corrected')
out.mkdir(parents=True, exist_ok=True)
with open(out / 'training_log.json', 'w') as f:
    json.dump(log, f, indent=2, default=str)
torch.save(model.state_dict(), out / 'model.pt')

print(f"\n{'='*60}")
print(f"  CARET v2 CORRECTED — {time.time()-start:.0f}s")
print(f"  train={train_ppl['ppl']:.1f} test={test_ppl['ppl']:.1f} gap={test_ppl['ppl']/train_ppl['ppl']:.3f}")
print(f"  crystal: 0={wstats['zero']:.3f} 1={wstats['one']:.3f} 3={wstats['three']:.3f}")
for band, c in children.items():
    print(f"  child[{band:>8s}]: 0={c['w0']:.3f} 1={c['w1']:.3f} 3={c['w3']:.3f}")
print(f"  v1 PPL: 76.7 | v2 PPL: {test_ppl['ppl']:.1f} | {'IMPROVED' if test_ppl['ppl'] < 76.7 else 'WORSE'}")
print(f"  1L prism PPL: 55.3 | v2 PPL: {test_ppl['ppl']:.1f}")
print(f"{'='*60}")
