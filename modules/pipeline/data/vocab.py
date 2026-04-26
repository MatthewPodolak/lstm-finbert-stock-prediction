import json
from pathlib import Path

def build_ticker_vocab(root_dir: Path, vocab_path: Path, timeframe_subdir: str) -> dict:
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    else:
        vocab = {}

    discovered = []
    for ticker_dir in sorted(root_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        if not (ticker_dir / timeframe_subdir).exists():
            continue
        discovered.append(ticker_dir.name)

    next_index = (max(vocab.values()) + 1) if vocab else 0

    for ticker_name in discovered:
        if ticker_name not in vocab:
            vocab[ticker_name] = next_index
            next_index += 1

    vocab_path.write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    return vocab