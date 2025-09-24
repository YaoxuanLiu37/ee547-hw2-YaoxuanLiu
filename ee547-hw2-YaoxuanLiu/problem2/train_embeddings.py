#!/usr/bin/env python3
import os, sys, json, re, argparse, random
from collections import Counter
from datetime import datetime, UTC
import torch
import torch.nn as nn
import torch.nn.functional as F

LETTER_RE = re.compile(r"[A-Za-z]+")
UNK_TOKEN = "<unk>"

def ts():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def info(msg):
    print(f"[{ts()}] INFO: {msg}")

def warn(msg):
    print(f"[{ts()}] WARNING: {msg}", file=sys.stderr)

def read_papers_json(papers_path):
    if not os.path.exists(papers_path):
        warn(f"papers.json not found: {papers_path}")
        return []
    with open(papers_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "papers" in data:
        data = data["papers"]
    if not isinstance(data, list):
        warn("papers.json top-level should be a list")
        return []
    return data

def clean_text(text):
    if not text:
        return []
    text = text.lower()
    tokens = LETTER_RE.findall(text)
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens

def docs_from_papers(papers):
    docs, ids = [], []
    for p in papers:
        abs_ = p.get("abstract") or ""
        toks = clean_text(abs_)
        docs.append(toks)
        ids.append(str(p.get("arxiv_id", "")))
    return docs, ids

def build_vocab(docs, max_vocab):
    counter = Counter()
    for toks in docs:
        counter.update(toks)
    most = counter.most_common(max_vocab)
    itos = [UNK_TOKEN] + [w for w, _ in most]
    stoi = {w: i for i, w in enumerate(itos)}
    return itos, stoi, counter

def make_sequences(docs, stoi, seq_len):
    seqs = []
    for toks in docs:
        ids = [stoi.get(t, 0) for t in toks]
        if len(ids) >= seq_len:
            ids = ids[:seq_len]
        else:
            ids = ids + [0] * (seq_len - len(ids))
        seqs.append(ids)
    return seqs

def make_bow_multi_hot_from_seqs(seqs, vocab_size, device):
    N, V = len(seqs), vocab_size
    X = torch.zeros((N, V), dtype=torch.float32, device=device)
    for i, ids in enumerate(seqs):
        idxs = set(j for j in ids if j != 0)
        if idxs:
            X[i, list(idxs)] = 1.0
    return X

class TiedAutoEncoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, sigmoid_out=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.W = nn.Parameter(torch.empty(vocab_size, emb_dim))
        self.b_enc = nn.Parameter(torch.zeros(emb_dim))
        self.b_dec = nn.Parameter(torch.zeros(vocab_size))
        nn.init.xavier_uniform_(self.W)
        self.sigmoid_out = sigmoid_out
    def encode(self, x):
        z = x @ self.W + self.b_enc
        return F.relu(z)
    def decode(self, z):
        y = z @ self.W.T + self.b_dec
        return torch.sigmoid(y) if self.sigmoid_out else y
    def forward(self, x):
        z = self.encode(x)
        yhat = self.decode(z)
        return yhat, z
    def word_embeddings(self):
        return self.W.detach().cpu()

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    start_time_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    info(f"Device: {device}")
    info(f"Loading abstracts from {args.papers}...")
    papers = read_papers_json(args.papers)
    if not papers:
        warn("No papers found; exiting.")
        sys.exit(1)
    docs, arxiv_ids = docs_from_papers(papers)
    info(f"Found {len(docs)} abstracts")
    itos, stoi, counter = build_vocab(docs, args.max_vocab)
    V = len(itos)
    info(f"Vocabulary size: {V} words (includes <unk>)")
    if V == 0:
        warn("Empty vocabulary; exiting.")
        sys.exit(1)
    seqs = make_sequences(docs, stoi, args.seq_len)
    os.makedirs(args.out, exist_ok=True)
    X = make_bow_multi_hot_from_seqs(seqs, V, device)
    has_rows = X.sum(dim=1) > 0
    if not bool(has_rows.any().item()):
        warn("All documents are empty after preprocessing; exiting.")
        sys.exit(1)
    X = X[has_rows]
    kept_ids = [arxiv_ids[i] for i, keep in enumerate(has_rows.tolist()) if keep]
    info(f"Model architecture: {V} → {args.emb_dim} → {V}")
    model = TiedAutoEncoder(vocab_size=V, emb_dim=args.emb_dim, sigmoid_out=True).to(device)
    n_params = count_params(model)
    info(f"Total parameters: {n_params:,}")
    if n_params > args.max_params:
        warn(f"Parameter limit exceeded: {n_params} > {args.max_params}. Reduce emb_dim or max_vocab.")
        sys.exit(1)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()
    N = X.size(0)
    order = list(range(N))
    info("Training autoencoder...")
    avg = None
    for epoch in range(1, args.epochs + 1):
        random.shuffle(order)
        total = 0.0
        model.train()
        for i in range(0, N, args.batch_size):
            idx = order[i:i + args.batch_size]
            xb = X[idx]
            yhat, _ = model(xb)
            loss = criterion(yhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        avg = total / N
        info(f"Epoch {epoch}/{args.epochs}, Loss: {avg:.6f}")

    vocab_to_idx = {w: i for i, w in enumerate(itos)}
    idx_to_vocab = {str(i): w for i, w in enumerate(itos)}
    vocab_size_topk = max(0, len(itos) - 1)
    hidden_dim = args.emb_dim
    embedding_dim = args.emb_dim

    model_path = os.path.join(args.out, "model.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_to_idx": vocab_to_idx,
        "model_config": {
            "vocab_size": vocab_size_topk,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim
        }
    }, model_path)

    with torch.no_grad():
        recon, Z_docs2 = model(X)
        row_loss = torch.nn.functional.binary_cross_entropy(recon, X, reduction="none").mean(dim=1).cpu().tolist()
    Z_cpu = Z_docs2.cpu().tolist()
    kept_index_map = {doc_id: i for i, doc_id in enumerate(kept_ids)}
    emb_list = []
    for arxid in arxiv_ids:
        if arxid in kept_index_map:
            i = kept_index_map[arxid]
            emb_list.append({
                "arxiv_id": arxid,
                "embedding": Z_cpu[i],
                "reconstruction_loss": float(row_loss[i])
            })
        else:
            emb_list.append({
                "arxiv_id": arxid,
                "embedding": [0.0] * embedding_dim,
                "reconstruction_loss": None
            })
    with open(os.path.join(args.out, "embeddings.json"), "w", encoding="utf-8") as f:
        json.dump(emb_list, f, ensure_ascii=False, indent=2)

    total_words_corpus = int(sum(counter.values()))
    vocab_json = {
        "vocab_to_idx": vocab_to_idx,
        "idx_to_vocab": idx_to_vocab,
        "vocab_size": vocab_size_topk,
        "total_words": total_words_corpus
    }
    with open(os.path.join(args.out, "vocabulary.json"), "w", encoding="utf-8") as f:
        json.dump(vocab_json, f, ensure_ascii=False, indent=2)

    end_time_utc = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    training_log = {
        "start_time": start_time_utc,
        "end_time": end_time_utc,
        "epochs": args.epochs,
        "final_loss": float(avg if avg is not None else 0.0),
        "total_parameters": int(n_params),
        "papers_processed": int(len(arxiv_ids)),
        "papers_used_for_training": int(X.size(0)),
        "embedding_dimension": int(embedding_dim)
    }
    with open(os.path.join(args.out, "training_log.json"), "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)
    info(f"Training complete. Artifacts saved to: {args.out}")
    info(json.dumps(training_log, ensure_ascii=False))

def parse_args():
    p = argparse.ArgumentParser(description="HW2 Problem2: Text Embedding Autoencoder with strict preprocessing")
    p.add_argument("input_papers", nargs="?", help="Path to papers.json (positional)")
    p.add_argument("output_dir", nargs="?", help="Output directory (positional)")
    default_papers = os.path.join(os.path.dirname(__file__), "..", "problem1", "sample_data", "papers.json")
    default_out = os.path.join(os.path.dirname(__file__), "outputs")
    p.add_argument("--papers", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--max-vocab", dest="max_vocab", type=int, default=5000)
    p.add_argument("--emb-dim", dest="emb_dim", type=int, default=256)
    p.add_argument("--seq-len", dest="seq_len", type=int, default=200)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--max-params", type=int, default=2_000_000)
    args = p.parse_args()
    args.papers = args.input_papers or args.papers or default_papers
    args.out = args.output_dir or args.out or default_out
    return args

if __name__ == "__main__":
    args = parse_args()
    train(args)
