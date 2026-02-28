"""
MigrateIQ — Model 1: Migration Route Predictor (LSTM)
======================================================
Predicts future GPS waypoints of an animal's migration path
given a sequence of historical positions + environmental variables.

Architecture: Encoder-Decoder LSTM with attention
Input:  Sequence of (lat, lon, day_of_year, temp, wind_speed, sst, ndvi)
Output: Next N predicted (lat, lon) waypoints with uncertainty bounds

Install: pip install torch numpy pandas scikit-learn
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import math


class MigrationDataset(Dataset):
    """
    Loads animal GPS tracking data.
    Each sample: (input_seq, target_seq)
      input_seq  → [seq_len, features]  : past positions + climate
      target_seq → [pred_len, 2]        : future (lat, lon)

    Expected CSV columns:
      animal_id, timestamp, lat, lon, temp_c, wind_ms, sst_c, ndvi
    """

    def __init__(self, df: pd.DataFrame, seq_len: int = 30, pred_len: int = 10):
        self.seq_len  = seq_len
        self.pred_len = pred_len

        # Feature columns
        self.feature_cols = ['lat', 'lon', 'day_of_year', 'temp_c', 'wind_ms', 'sst_c', 'ndvi']

        # Derived feature
        df = df.copy()
        df['timestamp']   = pd.to_datetime(df['timestamp'])
        df['day_of_year'] = df['timestamp'].dt.dayofyear / 365.0  # normalised

        # Normalize all features per-animal to handle different geographies
        self.scaler = MinMaxScaler()
        df[self.feature_cols] = self.scaler.fit_transform(df[self.feature_cols])

        # Build windows per animal
        self.samples = []
        for animal_id, group in df.groupby('animal_id'):
            group = group.sort_values('timestamp').reset_index(drop=True)
            values = group[self.feature_cols].values  # (T, F)
            coords = group[['lat', 'lon']].values     # (T, 2)

            total = seq_len + pred_len
            for i in range(len(values) - total + 1):
                x = values[i : i + seq_len]                    # (seq_len, F)
                y = coords[i + seq_len : i + seq_len + pred_len]  # (pred_len, 2)
                self.samples.append((
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(y, dtype=torch.float32)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



class BahdanauAttention(nn.Module):
    """Additive attention over encoder hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W_enc  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_dec  = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v      = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, enc_outputs, dec_hidden):
        """
        enc_outputs : (B, seq_len, H)
        dec_hidden  : (B, H)
        returns context (B, H), weights (B, seq_len)
        """
        dec_hidden  = dec_hidden.unsqueeze(1)                       # (B,1,H)
        energy      = torch.tanh(self.W_enc(enc_outputs) +
                                 self.W_dec(dec_hidden))            # (B,T,H)
        scores      = self.v(energy).squeeze(-1)                    # (B,T)
        weights     = torch.softmax(scores, dim=-1)                 # (B,T)
        context     = torch.bmm(weights.unsqueeze(1), enc_outputs)  # (B,1,H)
        return context.squeeze(1), weights


# ─────────────────────────────────────────────
# 3. ENCODER
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    """Bi-directional LSTM encoder for historical trajectory."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        # Project bidirectional output back to hidden_dim
        self.fc_h = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc_c = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        """
        x : (B, seq_len, input_dim)
        returns enc_out (B, seq_len, hidden_dim), (h,c) each (num_layers, B, H)
        """
        enc_out, (h, c) = self.lstm(x)

        # Merge forward + backward final hidden states
        # h shape: (num_layers*2, B, H) — take last layer
        h = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2H)
        c = torch.cat([c[-2], c[-1]], dim=1)
        h = torch.tanh(self.fc_h(h)).unsqueeze(0)  # (1, B, H)
        c = torch.tanh(self.fc_c(c)).unsqueeze(0)

        # Halve enc_out to hidden_dim
        B, T, _ = enc_out.shape
        enc_out = (enc_out[:,:,:enc_out.shape[-1]//2] +
                   enc_out[:,:,enc_out.shape[-1]//2:])  # simple sum

        return enc_out, (h, c)



class Decoder(nn.Module):
    """Autoregressive LSTM decoder with attention."""

    def __init__(self, hidden_dim: int, num_layers: int, dropout: float):
        super().__init__()
        # input = prev (lat, lon) + context vector
        self.lstm      = nn.LSTM(2 + hidden_dim, hidden_dim,
                                 num_layers=num_layers, batch_first=True,
                                 dropout=dropout if num_layers > 1 else 0.0)
        self.attention = BahdanauAttention(hidden_dim)
        self.fc_out    = nn.Linear(hidden_dim * 2, 2)    # predict (lat, lon)
        self.fc_sigma  = nn.Linear(hidden_dim * 2, 2)    # predict uncertainty

    def forward_step(self, prev_coord, enc_outputs, hidden):
        """
        One decoding step.
        prev_coord  : (B, 2)
        enc_outputs : (B, T, H)
        hidden      : tuple (h, c) each (layers, B, H)
        """
        h_last   = hidden[0][-1]                         # (B, H)
        context, attn_w = self.attention(enc_outputs, h_last)

        lstm_in  = torch.cat([prev_coord, context], dim=1).unsqueeze(1)  # (B,1,2+H)
        out, new_hidden = self.lstm(lstm_in, hidden)
        out      = out.squeeze(1)                        # (B, H)

        combined = torch.cat([out, context], dim=1)      # (B, 2H)
        coord    = self.fc_out(combined)                 # (B, 2) — mean
        sigma    = torch.exp(self.fc_sigma(combined))    # (B, 2) — std (always +)

        return coord, sigma, new_hidden, attn_w


class MigrationLSTM(nn.Module):
    """
    Encoder-Decoder LSTM for migration route prediction.

    Outputs:
      means  : (B, pred_len, 2)  — predicted (lat, lon)
      sigmas : (B, pred_len, 2)  — uncertainty (std dev per coord)
    """

    def __init__(
        self,
        input_dim:  int   = 7,      # number of input features
        hidden_dim: int   = 128,
        num_layers: int   = 2,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, num_layers, dropout)
        self.decoder = Decoder(hidden_dim, num_layers, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg=None, teacher_forcing_ratio: float = 0.5):
        """
        src  : (B, seq_len, input_dim)
        trg  : (B, pred_len, 2)  — ground truth coords (training only)
        """
        B = src.size(0)
        pred_len = trg.size(1) if trg is not None else 10

        enc_out, hidden = self.encoder(src)
        enc_out = self.dropout(enc_out)

        # Seed decoder with last known position
        prev_coord = src[:, -1, :2]   # last (lat, lon) from input

        means, sigmas = [], []
        for t in range(pred_len):
            coord, sigma, hidden, _ = self.decoder.forward_step(prev_coord, enc_out, hidden)
            means.append(coord.unsqueeze(1))
            sigmas.append(sigma.unsqueeze(1))

            # Teacher forcing during training
            if trg is not None and torch.rand(1).item() < teacher_forcing_ratio:
                prev_coord = trg[:, t, :]
            else:
                prev_coord = coord.detach()

        means  = torch.cat(means,  dim=1)   # (B, pred_len, 2)
        sigmas = torch.cat(sigmas, dim=1)   # (B, pred_len, 2)
        return means, sigmas



class GaussianNLLLoss(nn.Module):
    """
    Probabilistic loss: penalises both prediction error AND overconfidence.
    NLL = 0.5 * [log(sigma^2) + (y - mu)^2 / sigma^2]
    """
    def forward(self, means, sigmas, targets):
        var  = sigmas ** 2
        loss = 0.5 * (torch.log(var) + (targets - means) ** 2 / var)
        return loss.mean()



class RouteTrainer:
    def __init__(self, model, device='cpu', lr=1e-3):
        self.model     = model.to(device)
        self.device    = device
        self.criterion = GaussianNLLLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5, verbose=True
        )
        self.history   = {'train_loss': [], 'val_loss': []}

    def train_epoch(self, loader, teacher_forcing_ratio=0.5):
        self.model.train()
        total_loss = 0
        for src, trg in loader:
            src, trg = src.to(self.device), trg.to(self.device)
            self.optimizer.zero_grad()
            means, sigmas = self.model(src, trg, teacher_forcing_ratio)
            loss = self.criterion(means, sigmas, trg)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0
        for src, trg in loader:
            src, trg = src.to(self.device), trg.to(self.device)
            means, sigmas = self.model(src, trg, teacher_forcing_ratio=0.0)
            loss = self.criterion(means, sigmas, trg)
            total_loss += loss.item()
        return total_loss / len(loader)

    def fit(self, train_loader, val_loader, epochs=50, tf_decay=0.01):
        best_val  = float('inf')
        tf_ratio  = 0.5   # teacher forcing starts at 50%, decays each epoch

        for epoch in range(1, epochs + 1):
            tf_ratio  = max(0.0, tf_ratio - tf_decay)
            train_loss = self.train_epoch(train_loader, tf_ratio)
            val_loss   = self.eval_epoch(val_loader)
            self.scheduler.step(val_loss)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            print(f"Epoch {epoch:03d} | TF={tf_ratio:.2f} | "
                  f"Train={train_loss:.4f} | Val={val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(self.model.state_dict(), 'best_route_predictor.pt')
                print(f"           ✓ Saved best model (val={best_val:.4f})")

        return self.history



@torch.no_grad()
def predict_route(model, input_sequence: np.ndarray, pred_len: int = 20,
                  n_samples: int = 100, device: str = 'cpu'):
    """
    Monte-Carlo dropout inference for calibrated uncertainty.

    Args:
        model          : trained MigrationLSTM
        input_sequence : (seq_len, features) numpy array
        pred_len       : how many future steps to predict
        n_samples      : MC samples for uncertainty

    Returns:
        dict with:
          'mean'         : (pred_len, 2)  — best route
          'lower_95'     : (pred_len, 2)  — 95% CI lower bound
          'upper_95'     : (pred_len, 2)  — 95% CI upper bound
          'all_samples'  : (n_samples, pred_len, 2)
    """
    model.eval()
    # Enable dropout for MC sampling
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

    src = torch.tensor(input_sequence, dtype=torch.float32)\
               .unsqueeze(0).expand(n_samples, -1, -1).to(device)

    all_means = []
    for _ in range(n_samples):
        means, _ = model(src[:1], pred_len=pred_len)  # single sample
        all_means.append(means.squeeze(0).cpu().numpy())

    all_means = np.stack(all_means)  # (n_samples, pred_len, 2)
    mean      = all_means.mean(axis=0)
    lower_95  = np.percentile(all_means, 2.5,  axis=0)
    upper_95  = np.percentile(all_means, 97.5, axis=0)

    return {
        'mean':        mean,
        'lower_95':    lower_95,
        'upper_95':    upper_95,
        'all_samples': all_means
    }


# Synthetic data demo

def generate_synthetic_data(n_animals=20, timesteps=200):
    """Generates fake sinusoidal migration tracks for testing."""
    records = []
    for aid in range(n_animals):
        base_lat = np.random.uniform(-60, 70)
        base_lon = np.random.uniform(-180, 180)
        for t in range(timesteps):
            records.append({
                'animal_id':   f'animal_{aid:03d}',
                'timestamp':   pd.Timestamp('2015-01-01') + pd.Timedelta(days=t*2),
                'lat':         base_lat + 30 * math.sin(2 * math.pi * t / timesteps),
                'lon':         base_lon + 40 * math.sin(4 * math.pi * t / timesteps),
                'temp_c':      15 + 10 * math.sin(2 * math.pi * t / 365),
                'wind_ms':     np.random.uniform(0, 15),
                'sst_c':       20 + 5  * math.cos(2 * math.pi * t / 365),
                'ndvi':        np.random.uniform(0.2, 0.9),
            })
    return pd.DataFrame(records)


if __name__ == '__main__':
    print("=" * 60)
    print("MigrateIQ — Route Predictor (LSTM) — Quickstart")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Generate data
    df   = generate_synthetic_data(n_animals=30, timesteps=300)
    dataset = MigrationDataset(df, seq_len=30, pred_len=10)
    print(f"Dataset: {len(dataset)} training windows")

    # 2. Split
    train_size = int(0.8 * len(dataset))
    val_size   = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False, num_workers=0)

    # 3. Model
    model = MigrationLSTM(input_dim=7, hidden_dim=128, num_layers=2, dropout=0.2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # 4. Train
    trainer = RouteTrainer(model, device=device, lr=1e-3)
    history = trainer.fit(train_loader, val_loader, epochs=20)

    # 5. Inference
    sample_input = dataset[0][0].numpy()   # (seq_len, features)
    result = predict_route(model, sample_input, pred_len=10, n_samples=50)
    print("\nPrediction output:")
    print(f"  Mean route shape : {result['mean'].shape}")
    print(f"  95% CI lower     : {result['lower_95'].shape}")
    print(f"  Sample pt [0]    : lat={result['mean'][0,0]:.4f}, lon={result['mean'][0,1]:.4f}")
    print("\nDone. Model saved to best_route_predictor.pt")
