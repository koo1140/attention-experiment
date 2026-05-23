import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "dataset.json"
OUT_DIR = ROOT / "runs"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.set_num_threads(1)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_long_prefix(seed_prefix, mapping, raw, context_len, template_id):
    prefix = [tok.format(**mapping) for tok in seed_prefix]
    target_len = 7
    desired_prefix_len = max(len(prefix), context_len - target_len)

    rules = raw["vocab"]["rules"]
    items = raw["vocab"]["items"]
    fillers = raw["vocab"]["fillers"]
    steps = raw["vocab"]["steps"]

    decoy_rule = rules[(rules.index(mapping["rule_a"]) + template_id + 2) % len(rules)]
    if decoy_rule == mapping["rule_b"]:
        decoy_rule = rules[(rules.index(decoy_rule) + 1) % len(rules)]
    decoy_item = items[(items.index(mapping["item"]) + template_id + 3) % len(items)]

    noise_cycle = [
        steps[template_id % len(steps)],
        fillers[template_id % len(fillers)],
        decoy_rule,
        decoy_item,
        steps[(template_id + 3) % len(steps)],
        fillers[(template_id + 4) % len(fillers)],
    ]

    cursor = 0
    while len(prefix) < desired_prefix_len:
        prefix.append(noise_cycle[cursor % len(noise_cycle)])
        cursor += 1
    return prefix[:desired_prefix_len]


def expand_templates(raw, split, context_len, limit=None):
    rules = raw["vocab"]["rules"]
    items = raw["vocab"]["items"]
    templates = raw[f"{split}_templates"]
    examples = []

    for item in items:
        for rule_a in rules:
            for rule_b in rules:
                if rule_a == rule_b:
                    continue
                for template_id, template in enumerate(templates):
                    mapping = {"rule_a": rule_a, "rule_b": rule_b, "item": item}
                    prefix = build_long_prefix(
                        template["prefix"], mapping, raw, context_len, template_id
                    )
                    target = [tok.format(**mapping) for tok in template["target"]]
                    examples.append(
                        {
                            "id": f"{split}-{template_id}-{item}-{rule_a}-{rule_b}",
                            "prefix": prefix,
                            "target": target,
                            "rule_a": rule_a,
                            "rule_b": rule_b,
                            "item": item,
                        }
                    )

    random.shuffle(examples)
    return examples if limit is None else examples[:limit]


def load_dataset(context_len, train_limit=800, val_limit=180):
    raw = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    words = ["<pad>", "<bos>", "<eos>"]
    for group in ("rules", "items", "steps", "fillers"):
        words.extend(raw["vocab"][group])

    vocab = {word: i for i, word in enumerate(dict.fromkeys(words))}
    inv_vocab = {i: word for word, i in vocab.items()}
    train = expand_templates(raw, "train", context_len, train_limit)
    val = expand_templates(raw, "validation", context_len, val_limit)
    meta = {
        "rules": set(raw["vocab"]["rules"]),
        "items": set(raw["vocab"]["items"]),
    }
    return vocab, inv_vocab, train, val, meta


def encode_example(example, vocab):
    seq = example["prefix"] + example["target"]
    ids = torch.tensor([vocab[tok] for tok in seq], dtype=torch.long)
    prefix_len = len(example["prefix"])
    labels = ids.clone()
    labels[:prefix_len] = -100
    labels = torch.cat([labels[1:], torch.tensor([-100])])
    return ids, labels


def make_batch(examples, vocab, batch_size):
    batch = random.sample(examples, batch_size)
    encoded = [encode_example(ex, vocab) for ex in batch]
    max_len = max(len(ids) for ids, _ in encoded)
    x = torch.full((batch_size, max_len), vocab["<pad>"], dtype=torch.long)
    y = torch.full((batch_size, max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(encoded):
        x[i, : len(ids)] = ids
        y[i, : len(labels)] = labels
    return x, y


class TinyAttentionLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, n_layers=2, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        bsz, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.tok(x) + self.pos(positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        h = self.blocks(h, mask=mask, is_causal=True)
        return self.head(self.norm(h))


class CheapContextBlock(nn.Module):
    def __init__(self, d_model, slots=8):
        super().__init__()
        self.slots = slots
        self.in_proj = nn.Linear(d_model, d_model * 3)
        self.mix = nn.Linear(d_model * 2, d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h):
        bsz, seq_len, dim = h.shape
        memory = h.new_zeros(bsz, self.slots, dim)
        outputs = []

        for t in range(seq_len):
            x_t = h[:, t]
            gate_raw, write_raw, read_raw = self.in_proj(x_t).chunk(3, dim=-1)
            gate = torch.sigmoid(gate_raw).unsqueeze(1)
            write = torch.tanh(write_raw).unsqueeze(1)

            # Deterministic rotating write pattern: no token classification, just cheap implicit slots.
            slot = t % self.slots
            slot_mask = h.new_zeros(1, self.slots, 1)
            slot_mask[:, slot : slot + 1] = 1.0
            updated_slot = memory + slot_mask * gate * (write - memory)
            memory = memory * (1.0 - slot_mask) + updated_slot * slot_mask

            read = torch.tanh(read_raw).unsqueeze(1)
            score = (memory * read).sum(dim=-1) / math.sqrt(dim)
            weights = torch.softmax(score, dim=-1).unsqueeze(-1)
            ctx = (weights * memory).sum(dim=1)
            y = self.mix(torch.cat([x_t, ctx], dim=-1))
            outputs.append(y)

        y = torch.stack(outputs, dim=1)
        return self.norm(h + y + self.ff(h + y))


class TinyCompressedLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, slots=8, max_len=64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([CheapContextBlock(d_model, slots) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        bsz, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)
        h = self.tok(x) + self.pos(positions)
        for block in self.blocks:
            h = block(h)
        return self.head(self.norm(h))


@torch.no_grad()
def evaluate(model, examples, vocab, meta, device, batch_size=64):
    model.eval()
    losses = []
    correct = 0
    total = 0
    rule_correct = 0
    rule_total = 0
    item_correct = 0
    item_total = 0
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        x, y = make_fixed_batch(batch, vocab)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        losses.append(loss.item())
        pred = logits.argmax(dim=-1)
        mask = y.ne(-100)
        correct += pred[mask].eq(y[mask]).sum().item()
        total += mask.sum().item()
        rule_mask = torch.zeros_like(mask)
        item_mask = torch.zeros_like(mask)
        for row, example in enumerate(batch):
            seq = example["prefix"] + example["target"]
            prefix_len = len(example["prefix"])
            for pos in range(prefix_len, len(seq) - 1):
                tok = seq[pos]
                if tok in meta["rules"]:
                    rule_mask[row, pos - 1] = True
                if tok in meta["items"]:
                    item_mask[row, pos - 1] = True
        rule_correct += pred[rule_mask].eq(y[rule_mask]).sum().item()
        rule_total += rule_mask.sum().item()
        item_correct += pred[item_mask].eq(y[item_mask]).sum().item()
        item_total += item_mask.sum().item()
    return {
        "loss": sum(losses) / len(losses),
        "token_acc": correct / max(total, 1),
        "rule_acc": rule_correct / max(rule_total, 1),
        "item_acc": item_correct / max(item_total, 1),
    }


def make_fixed_batch(examples, vocab):
    encoded = [encode_example(ex, vocab) for ex in examples]
    max_len = max(len(ids) for ids, _ in encoded)
    x = torch.full((len(examples), max_len), vocab["<pad>"], dtype=torch.long)
    y = torch.full((len(examples), max_len), -100, dtype=torch.long)
    for i, (ids, labels) in enumerate(encoded):
        x[i, : len(ids)] = ids
        y[i, : len(labels)] = labels
    return x, y


def train_one(benchmark, name, model, train, val, vocab, meta, device, steps, batch_size, lr):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    history = []
    start_time = time.perf_counter()

    for step in range(1, steps + 1):
        model.train()
        x, y = make_batch(train, vocab, batch_size)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 25 == 0 or step == steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            metrics = evaluate(model, val, vocab, meta, device)
            elapsed = time.perf_counter() - start_time
            history.append(
                {
                    "benchmark": benchmark,
                    "model": name,
                    "step": step,
                    "train_loss": loss.item(),
                    "val_loss": metrics["loss"],
                    "val_acc": metrics["token_acc"],
                    "rule_acc": metrics["rule_acc"],
                    "item_acc": metrics["item_acc"],
                    "elapsed_sec": elapsed,
                }
            )
            print(
                f"{benchmark:10s} {name:11s} step={step:4d} train_loss={loss.item():.4f} "
                f"val_loss={metrics['loss']:.4f} val_acc={metrics['token_acc']:.3f} "
                f"rule_acc={metrics['rule_acc']:.3f} elapsed={elapsed:.1f}s"
            )

    return history


@torch.no_grad()
def sample_predictions(model, examples, vocab, inv_vocab, device, count=5):
    model.eval()
    rows = []
    for example in examples[:count]:
        ids, labels = encode_example(example, vocab)
        logits = model(ids.unsqueeze(0).to(device))
        preds = logits.argmax(dim=-1).squeeze(0)
        mask = labels.ne(-100).to(device)
        rows.append(
            {
                "prefix": " ".join(example["prefix"]),
                "target": " ".join(example["target"]),
                "pred": " ".join(inv_vocab[int(tok)] for tok in preds[mask].detach().cpu()),
            }
        )
    return rows


def plot_history(history, out_dir):
    out_dir.mkdir(exist_ok=True)
    labels = sorted({(row["benchmark"], row["model"]) for row in history})

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    chart_specs = [
        ("val_loss", "Validation loss", axes[0, 0], None),
        ("val_acc", "Validation token accuracy", axes[0, 1], (0, 1.02)),
        ("rule_acc", "Rule retention accuracy", axes[1, 0], (0, 1.02)),
    ]

    for metric, title, ax, ylim in chart_specs:
        for benchmark, model in labels:
            rows = [
                row for row in history if row["benchmark"] == benchmark and row["model"] == model
            ]
            ax.plot(
                [r["step"] for r in rows],
                [r[metric] for r in rows],
                marker="o",
                label=f"{model}@{benchmark}",
            )
        ax.set_title(title)
        ax.set_xlabel("Training step")
        if metric == "val_loss":
            ax.set_ylabel("Cross entropy")
        else:
            ax.set_ylabel("Accuracy")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.2)

    final_rows = []
    for benchmark, model in labels:
        rows = [row for row in history if row["benchmark"] == benchmark and row["model"] == model]
        final_rows.append((f"{model}@{benchmark}", rows[-1]["elapsed_sec"]))
    axes[1, 1].bar([x[0] for x in final_rows], [x[1] for x in final_rows])
    axes[1, 1].set_title("CPU training time")
    axes[1, 1].set_ylabel("Seconds")
    axes[1, 1].tick_params(axis="x", rotation=35)

    handles, labels_text = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_text, loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_dir / "benchmark_overview.png", dpi=160)
    plt.close(fig)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--contexts", type=str, default="64,256,1028")
    parser.add_argument("--train-limit", type=int, default=480)
    parser.add_argument("--val-limit", type=int, default=120)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device()
    print(f"device={device}")
    context_sizes = [int(part.strip()) for part in args.contexts.split(",") if part.strip()]
    history = []
    report = {}

    for context_len in context_sizes:
        benchmark = f"ctx{context_len}"
        vocab, inv_vocab, train, val, meta = load_dataset(
            context_len=context_len,
            train_limit=args.train_limit,
            val_limit=args.val_limit,
        )
        seq_len = max(len(ex["prefix"]) + len(ex["target"]) for ex in train + val)
        models = {
            "attention": TinyAttentionLM(len(vocab), max_len=seq_len),
            "compressed": TinyCompressedLM(len(vocab), max_len=seq_len),
        }

        print(
            f"{benchmark:10s} train_examples={len(train)} val_examples={len(val)} "
            f"vocab_size={len(vocab)} seq_len={seq_len}"
        )
        for name, model in models.items():
            model = model.to(device)
            print(f"{benchmark:10s} {name:11s} parameters={parameter_count(model):,}")
            history.extend(
                train_one(
                    benchmark,
                    name,
                    model,
                    train,
                    val,
                    vocab,
                    meta,
                    device,
                    args.steps,
                    args.batch_size,
                    args.lr,
                )
            )
        report[benchmark] = {
            name: sample_predictions(model, val, vocab, inv_vocab, device)
            for name, model in models.items()
        }

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    plot_history(history, OUT_DIR)
    (OUT_DIR / "predictions.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"wrote {OUT_DIR / 'history.json'}")
    print(f"wrote {OUT_DIR / 'predictions.json'}")
    print(f"wrote {OUT_DIR / 'benchmark_overview.png'}")


if __name__ == "__main__":
    main()
