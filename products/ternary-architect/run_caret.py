#!/usr/bin/env python3
"""CARET Figure 14.15 transcoded — run experiment."""
import json, time, sys, numpy as np, torch
from pathlib import Path
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "topological-router"))

from ternary_linear import TernaryLinear
from run_long import TextChunked, split_dataset, measure_perplexity, weight_stats
from topo_measures import effective_rank, spectral_gap, gini_fast

# Import the CARET model inline (same code as verification above)
import torch.nn as nn
from ternary_linear import make_linear
from ternary_transformer import GPTConfig, Block

class CARETTranscoded(nn.Module):
    def __init__(self, vocab_size=50257, n_embd=2048, n_head=16, block_size=512):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(0.0)
        self.block_size = block_size
        
        gpt_cfg = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                            n_layer=4, n_head=n_head, n_embd=n_embd,
                            dropout=0.0, weight_set='013')
        self.parent_rings = nn.ModuleList([Block(gpt_cfg) for _ in range(4)])
        self.ln_parent = nn.LayerNorm(n_embd)
        
        n_prime = int(n_embd * 0.50)
        n_ident = int(n_embd * 0.33)
        n_void  = n_embd - n_prime - n_ident
        self.split_sizes = [n_void, n_ident, n_prime]
        
        self.child_ident = nn.Sequential(
            make_linear(n_ident, n_ident, bias=False, weight_set='013'),
            nn.LayerNorm(n_ident), nn.GELU(),
            make_linear(n_ident, n_ident, bias=False, weight_set='013'),
            nn.LayerNorm(n_ident), nn.GELU(),
            make_linear(n_ident, n_ident, bias=False, weight_set='013'),
            nn.LayerNorm(n_ident))
        self.child_prime = nn.Sequential(
            make_linear(n_prime, n_prime, bias=False, weight_set='013'),
            nn.LayerNorm(n_prime), nn.GELU(),
            make_linear(n_prime, n_prime, bias=False, weight_set='013'),
            nn.LayerNorm(n_prime), nn.GELU(),
            make_linear(n_prime, n_prime, bias=False, weight_set='013'),
            nn.LayerNorm(n_prime))
        self.child_void = nn.Sequential(
            make_linear(n_void, n_void, bias=False, weight_set='013'),
            nn.LayerNorm(n_void), nn.GELU(),
            make_linear(n_void, n_void, bias=False, weight_set='013'),
            nn.LayerNorm(n_void), nn.GELU(),
            make_linear(n_void, n_void, bias=False, weight_set='013'),
            nn.LayerNorm(n_void))
        
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
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
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        hs = []
        for ring in self.parent_rings:
            x = ring(x)
            if return_hidden: hs.append(x.detach())
        x = self.ln_parent(x)
        v, i, p = torch.split(x, self.split_sizes, dim=-1)
        v = self.child_void(v); i = self.child_ident(i); p = self.child_prime(p)
        x = torch.cat([v, i, p], dim=-1)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                                targets.view(-1), ignore_index=-100)
        return (logits, loss, hs) if return_hidden else (logits, loss)

def per_child_crystal(model):
    results = {}
    for name, child in [('void', model.child_void), ('identity', model.child_ident), ('prime', model.child_prime)]:
        w0, w1, w3, total = 0, 0, 0, 0
        for m in child.modules():
            if isinstance(m, TernaryLinear):
                wq = m.get_quantized_weight()
                n = wq.numel()
                w0 += (wq == 0).sum().item()
                w1 += (wq == 1).sum().item()
                w3 += (wq == 3).sum().item()
                total += n
        if total > 0:
            results[name] = {'w0': w0/total, 'w1': w1/total, 'w3': w3/total}
    return results

# --- Main ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

ds = TextChunked(tokenizer, max_length=256, n_samples=200000, dataset_name="wikitext")
train_ds, test_ds = split_dataset(ds)
train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)
print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

model = CARETTranscoded(vocab_size=tokenizer.vocab_size).to(device)
params = sum(p.numel() for p in model.parameters())
print(f"{params:,} params")

ternary_p = [p for n, p in model.named_parameters() if 'tok_emb' not in n and 'pos_emb' not in n and 'ln' not in n and 'lm_head' not in n]
continuous_p = [p for n, p in model.named_parameters() if any(k in n for k in ['tok_emb', 'pos_emb', 'ln', 'lm_head'])]
optimizer = torch.optim.AdamW([
    {'params': ternary_p, 'weight_decay': 0.0},  # ZERO L2 — measure the TRUE crystal
    {'params': continuous_p, 'weight_decay': 0.01}
], lr=3e-4)

warmup = 2000; max_steps = 20000; accum = 4
def lr_lambda(step):
    if step < warmup: return step / max(1, warmup)
    progress = (step - warmup) / max(1, max_steps - warmup)
    return 0.05 + 0.95 * (1 + np.cos(np.pi * progress)) / 2
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

log = {"experiment": "CARET_14.15_transcoded", "checkpoints": [], "steps": []}
step = 0; start = time.time()
print(f"\nCARET Figure 14.15 — 20K steps, L2=0.0 on ternary\n")
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
            print(f"  {step:6d} | loss {loss.item():.4f} | ppl {ppl:.1f} | {time.time()-start:.0f}s", flush=True)
        if step > 0 and step % 1000 == 0:
            train_ppl = measure_perplexity(model, train_loader, device)
            test_ppl = measure_perplexity(model, test_loader, device)
            wstats = weight_stats(model)
            children = per_child_crystal(model)
            gap = test_ppl['ppl'] / train_ppl['ppl']
            ckpt = {'step': step, 'train_ppl': train_ppl, 'test_ppl': test_ppl,
                    'gen_gap': gap, 'weight_dist': wstats, 'child_crystals': children}
            log['checkpoints'].append(ckpt)
            print(f"\n    CKPT {step:6d} | train {train_ppl['ppl']:.1f} | test {test_ppl['ppl']:.1f} | gap {gap:.3f}")
            print(f"    weights: 0={wstats['zero']:.3f} 1={wstats['one']:.3f} 3={wstats['three']:.3f}")
            for band, c in children.items():
                print(f"    child[{band:>8s}]: 0={c['w0']:.3f} 1={c['w1']:.3f} 3={c['w3']:.3f}")
            print(flush=True)
        step += 1

# Final
train_ppl = measure_perplexity(model, train_loader, device)
test_ppl = measure_perplexity(model, test_loader, device)
wstats = weight_stats(model)
children = per_child_crystal(model)
print(f"\n{'='*60}")
print(f"  CARET 14.15 TRANSCODED — {time.time()-start:.0f}s")
print(f"  train={train_ppl['ppl']:.1f} test={test_ppl['ppl']:.1f} gap={test_ppl['ppl']/train_ppl['ppl']:.3f}")
print(f"  crystal: 0={wstats['zero']:.3f} 1={wstats['one']:.3f} 3={wstats['three']:.3f}")
for band, c in children.items():
    print(f"  child[{band:>8s}]: 0={c['w0']:.3f} 1={c['w1']:.3f} 3={c['w3']:.3f}")
print(f"{'='*60}")

out = Path('results/caret_14_15_transcoded')
out.mkdir(parents=True, exist_ok=True)
log['final'] = {'train_ppl': train_ppl, 'test_ppl': test_ppl, 'weight_dist': wstats, 'child_crystals': children}
with open(out / 'training_log.json', 'w') as f: json.dump(log, f, indent=2, default=str)
torch.save(model.state_dict(), out / 'model.pt')
