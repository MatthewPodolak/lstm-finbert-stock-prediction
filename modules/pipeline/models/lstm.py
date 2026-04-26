import torch
import torch.nn as nn

class ModernLstmClassifier(nn.Module):
    def __init__(self, input_dim, num_tickers, hidden_dim=128, num_layers=2, dropout=0.2, num_classes=3, ticker_emb_dim=16):
        super().__init__()

        self.ticker_emb = nn.Embedding(num_tickers, ticker_emb_dim)

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim + ticker_emb_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )

        self.cls = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, features, ticker_idx):
        emb = self.ticker_emb(ticker_idx).unsqueeze(1).expand(-1, features.size(1), -1)
        x = self.input_projection(torch.cat([features, emb], dim=2))
        out, _ = self.lstm(x)
        return self.cls(out[:, -1, :])
