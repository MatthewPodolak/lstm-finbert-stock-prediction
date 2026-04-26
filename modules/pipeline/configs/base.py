from pathlib import Path
import torch

ROOT_DIR = Path(r"C:\Users\RudyChemik\Desktop\stock_data")
NEWS_DIR = Path(r"C:\Users\RudyChemik\Desktop\stock_news")

OUTPUT_DIR = Path("./bin")
OUTPUT_DIR.mkdir(exist_ok=True)

VOCAB_PATH = OUTPUT_DIR / "ticker_vocab.json"

HIDDEN_DIM = 128
NUM_LAYERS = 2
TICKER_EMB_DIM = 16
NUM_CLASSES = 3

BATCH_SIZE = 512
EPOCHS = 50
LR = 5e-4
DROPOUT = 0.2
PATIENCE = 1
SEED = 2115

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")