import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class FinBERTAnalyzer:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.eval()

    @torch.no_grad()
    def analyze(self, text: str) -> dict:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        probs = torch.softmax(self.model(**tokens).logits, dim=1)[0]
        score = probs[0].item() * 1.0 + probs[2].item() * 0.5
        return {"score": round(score, 4)}
