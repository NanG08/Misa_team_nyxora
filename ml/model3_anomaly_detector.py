"""
MigrateIQ — Model 3: Migration Anomaly Detector
================================================
Detects when a migration season is behaving unusually compared to
historical baselines. Triggers the "Why It's Changing" story cards.

What it detects:
  - Route deviation  : path is significantly different from historical mean
  - Timing anomaly   : migration started earlier/later than expected
  - Speed anomaly    : animals moving much faster/slower than normal
  - Population gap   : tracking density far below historical average

Architecture: LSTM Autoencoder (unsupervised)
  - Trained only on NORMAL seasons
  - High reconstruction error → anomaly
  - Outputs: anomaly score, anomaly type, deviation vector

Install: pip install torch numpy pandas scikit-learn
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
import math



@dataclass
class AnomalyResult:
    """Output of anomaly detection for one migration season."""
    species:        str
    season_year:    int
    anomaly_score:  float           # 0–1, higher = more anomalous
    is_anomaly:     bool
    anomaly_type:   list[str]       # e.g. ['route_deviation', 'timing']
    deviation_vec:  np.ndarray      # per-feature deviation
    timing_shift_days: Optional[float] = None
    route_deviation_km: Optional[float] = None
    severity:       str = 'normal'  # 'normal' | 'mild' | 'moderate' | 'severe'
    story_trigger:  Optional[str] = None   # text for the story feed


def extract_season_features(df: pd.DataFrame, species: str,
                             year: int, seq_len: int = 60) -> np.ndarray:
    """
    Converts raw GPS + climate data for one species/year into
    a fixed-length feature sequence for the autoencoder.

    Features per timestep:
      lat_norm, lon_norm, speed_kmh, bearing_deg_sin, bearing_deg_cos,
      temp_anomaly, ndvi_anomaly, days_from_start

    Returns: (seq_len, 8) array — zero-padded/truncated to seq_len
    """
    subset = df[(df['species'] == species) & (df['year'] == year)].copy()
    subset = subset.sort_values('day_of_year').reset_index(drop=True)

    if len(subset) < 5:
        return np.zeros((seq_len, 8))

    # Compute derived features
    lats  = subset['lat'].values
    lons  = subset['lon'].values

    # Speed (simplified Euclidean, replace with Haversine in production)
    dlat  = np.diff(lats, prepend=lats[0])
    dlon  = np.diff(lons, prepend=lons[0])
    speed = np.sqrt(dlat**2 + dlon**2) * 111     # rough km/deg

    # Bearing
    bearing = np.arctan2(dlon, dlat)

    # Normalise positions
    lat_norm  = (lats - lats.mean()) / (lats.std() + 1e-6)
    lon_norm  = (lons - lons.mean()) / (lons.std() + 1e-6)

    # Temperature anomaly (deviation from mean)
    temp_mean = subset['temp_c'].mean()
    temp_anom = subset['temp_c'] - temp_mean

    ndvi_mean = subset['ndvi'].mean()
    ndvi_anom = subset['ndvi'] - ndvi_mean

    days = subset['day_of_year'].values
    days_norm = (days - days.min()) / max(days.max() - days.min(), 1)

    features = np.stack([
        lat_norm, lon_norm, speed,
        np.sin(bearing), np.cos(bearing),
        temp_anom, ndvi_anom, days_norm
    ], axis=1)   # (T, 8)

    # Pad or truncate to seq_len
    T = len(features)
    if T >= seq_len:
        return features[:seq_len]
    else:
        pad = np.zeros((seq_len - T, 8))
        return np.vstack([features, pad])



class SeasonDataset(Dataset):
    """
    Each sample is one migration season's feature sequence.
    Only NORMAL seasons are used during training.
    """

    def __init__(self, sequences: list[np.ndarray]):
        self.sequences = [
            torch.tensor(s, dtype=torch.float32) for s in sequences
        ]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]



class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc   = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        z = self.fc(h[-1])   # (B, latent_dim)
        return z, out


class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, seq_len, num_layers, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.fc_init = nn.Linear(latent_dim, hidden_dim)
        self.lstm    = nn.LSTM(latent_dim, hidden_dim, num_layers=num_layers,
                               batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.fc_out  = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # Repeat latent vector for each timestep
        repeated = z.unsqueeze(1).repeat(1, self.seq_len, 1)  # (B, T, latent)
        out, _   = self.lstm(repeated)
        recon    = self.fc_out(out)                            # (B, T, features)
        return recon


class MigrationAutoencoder(nn.Module):
    """
    LSTM Autoencoder.
    Trained to reconstruct NORMAL migration seasons.
    High reconstruction error = anomaly.
    """

    def __init__(
        self,
        input_dim:   int   = 8,
        hidden_dim:  int   = 64,
        latent_dim:  int   = 16,
        seq_len:     int   = 60,
        num_layers:  int   = 2,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.encoder = LSTMEncoder(input_dim, hidden_dim, latent_dim, num_layers, dropout)
        self.decoder = LSTMDecoder(latent_dim, hidden_dim, input_dim, seq_len, num_layers, dropout)
        self.latent_dim = latent_dim

    def forward(self, x):
        z, enc_out = self.encoder(x)
        recon      = self.decoder(z)
        return recon, z

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Returns per-sample, per-timestep MSE. Shape: (B, T, F)"""
        with torch.no_grad():
            recon, _ = self.forward(x)
            return (recon - x) ** 2


class AnomalyTrainer:
    def __init__(self, model: MigrationAutoencoder, device='cpu', lr=1e-3):
        self.model     = model.to(device)
        self.device    = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.threshold = None    # set after training via calibration

    def train_epoch(self, loader):
        self.model.train()
        total = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            recon, _ = self.model(batch)
            loss = self.criterion(recon, batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            total += loss.item()
        return total / len(loader)

    def fit(self, train_loader, val_loader=None, epochs=40):
        for epoch in range(1, epochs + 1):
            loss = self.train_epoch(train_loader)
            val_str = ''
            if val_loader:
                val_loss = self._eval(val_loader)
                val_str = f' | Val={val_loss:.4f}'
            print(f"Epoch {epoch:03d} | Train={loss:.4f}{val_str}")
        torch.save(self.model.state_dict(), 'anomaly_detector.pt')

    @torch.no_grad()
    def _eval(self, loader):
        self.model.eval()
        total = 0
        for batch in loader:
            batch = batch.to(self.device)
            recon, _ = self.model(batch)
            total += nn.functional.mse_loss(recon, batch).item()
        return total / len(loader)

    def calibrate_threshold(self, normal_sequences: list[np.ndarray],
                            percentile: float = 95.0):
        """
        Sets the anomaly threshold by computing reconstruction errors
        on known-normal data and taking the Nth percentile.
        Seasons above this threshold are flagged as anomalous.
        """
        self.model.eval()
        errors = []
        with torch.no_grad():
            for seq in normal_sequences:
                x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                err = self.model.reconstruction_error(x)
                errors.append(err.mean().item())

        self.threshold = float(np.percentile(errors, percentile))
        print(f"Anomaly threshold set @ {percentile}th percentile = {self.threshold:.6f}")
        return self.threshold



class AnomalyDetector:
    """
    High-level interface: given a migration season's data,
    returns structured AnomalyResult with score, type, and story trigger.
    """

    def __init__(self, model: MigrationAutoencoder, threshold: float,
                 device: str = 'cpu'):
        self.model     = model.to(device)
        self.device    = device
        self.threshold = threshold
        self.model.eval()

        # Historical baseline per species (mean + std of seasonal stats)
        # In production: load from a precomputed database
        self.baselines: dict = defaultdict(lambda: {
            'start_doy_mean': 90,  'start_doy_std': 7,
            'route_mean_lat': 45,  'route_std_lat': 3,
            'speed_mean': 120,     'speed_std': 20,
        })

    @torch.no_grad()
    def detect(self, sequence: np.ndarray, species: str,
               season_year: int, season_df: pd.DataFrame = None) -> AnomalyResult:
        """
        Main detection method.
        sequence : (seq_len, features) array from extract_season_features()
        """
        x   = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)
        err = self.model.reconstruction_error(x)  # (1, T, F)
        recon_error = err.mean().item()

        # Per-feature error (deviation vector)
        feat_errors = err.squeeze(0).mean(dim=0).cpu().numpy()  # (F,)

        # Normalise to 0-1 score (sigmoid around threshold)
        score = 1 / (1 + math.exp(-10 * (recon_error - self.threshold)))

        is_anomaly   = recon_error > self.threshold
        anomaly_type = []
        baseline     = self.baselines[species]

        # ── Timing anomaly ─────────────────────────
        timing_shift = None
        if season_df is not None and 'day_of_year' in season_df.columns:
            actual_start = season_df['day_of_year'].min()
            timing_shift = actual_start - baseline['start_doy_mean']
            if abs(timing_shift) > baseline['start_doy_std'] * 2:
                anomaly_type.append('timing_shift')

        # ── Route deviation ─────────────────────────
        route_dev_km = None
        if season_df is not None and 'lat' in season_df.columns:
            actual_lat  = season_df['lat'].mean()
            route_dev   = abs(actual_lat - baseline['route_mean_lat'])
            route_dev_km = route_dev * 111
            if route_dev > baseline['route_std_lat'] * 2:
                anomaly_type.append('route_deviation')

        # ── Reconstruction-based types ──────────────
        feature_names = ['lat','lon','speed','bearing_sin','bearing_cos',
                         'temp_anom','ndvi_anom','days']
        top_feat_idx  = feat_errors.argsort()[-2:]
        for idx in top_feat_idx:
            if feat_errors[idx] > self.threshold * 0.5:
                fname = feature_names[idx] if idx < len(feature_names) else f'feat_{idx}'
                if 'speed' in fname:       anomaly_type.append('speed_anomaly')
                elif 'temp' in fname:      anomaly_type.append('climate_correlation')
                elif 'ndvi' in fname:      anomaly_type.append('habitat_degradation')

        anomaly_type = list(set(anomaly_type))

        # ── Severity ────────────────────────────────
        if   score > 0.85:   severity = 'severe'
        elif score > 0.65:   severity = 'moderate'
        elif score > 0.45:   severity = 'mild'
        else:                severity = 'normal'

        # ── Story trigger ────────────────────────────
        story_trigger = self._build_story_trigger(
            species, season_year, anomaly_type,
            timing_shift, route_dev_km, severity
        )

        return AnomalyResult(
            species           = species,
            season_year       = season_year,
            anomaly_score     = score,
            is_anomaly        = is_anomaly,
            anomaly_type      = anomaly_type,
            deviation_vec     = feat_errors,
            timing_shift_days = timing_shift,
            route_deviation_km= route_dev_km,
            severity          = severity,
            story_trigger     = story_trigger
        )

    def _build_story_trigger(self, species, year, types,
                              timing_shift, route_dev, severity):
        if severity == 'normal':
            return None

        parts = [f"{species} — {year} season flagged as {severity.upper()} anomaly."]
        if 'timing_shift' in types and timing_shift:
            direction = 'earlier' if timing_shift < 0 else 'later'
            parts.append(f"Migration started {abs(timing_shift):.0f} days {direction} than historical average.")
        if 'route_deviation' in types and route_dev:
            parts.append(f"Route deviated {route_dev:.0f} km from expected corridor.")
        if 'speed_anomaly' in types:
            parts.append("Unusual travel speed detected — possible environmental barrier.")
        if 'habitat_degradation' in types:
            parts.append("NDVI drop detected along route — vegetation stress likely.")
        if 'climate_correlation' in types:
            parts.append("Temperature anomaly correlated with route change.")

        return ' '.join(parts)



def generate_season_sequences(n_normal=100, n_anomalous=20, seq_len=60):
    """Generates normal + anomalous migration season feature sequences."""
    rng = np.random.default_rng(42)
    normal, anomalous = [], []

    for _ in range(n_normal):
        t = np.linspace(0, 1, seq_len)
        seq = np.stack([
            np.sin(2*np.pi*t) + rng.normal(0, 0.1, seq_len),   # lat
            np.cos(2*np.pi*t) + rng.normal(0, 0.1, seq_len),   # lon
            np.ones(seq_len)*2 + rng.normal(0, 0.2, seq_len),  # speed
            np.sin(np.pi*t)    + rng.normal(0, 0.05, seq_len), # bearing_sin
            np.cos(np.pi*t)    + rng.normal(0, 0.05, seq_len), # bearing_cos
            rng.normal(0, 1, seq_len),  # temp_anom
            rng.normal(0, 0.5, seq_len),# ndvi_anom
            t                           # days_norm
        ], axis=1)
        normal.append(seq)

    for _ in range(n_anomalous):
        t = np.linspace(0, 1, seq_len)
        # Inject anomaly: abrupt route shift + timing offset
        shift_point = rng.integers(20, 40)
        seq = np.stack([
            np.sin(2*np.pi*t) + rng.normal(0, 0.1, seq_len) +
                np.where(np.arange(seq_len) > shift_point, 2.5, 0),  # ← deviation
            np.cos(2*np.pi*t) + rng.normal(0, 0.1, seq_len),
            np.ones(seq_len)*2 + rng.normal(0, 0.5, seq_len) +
                np.where(np.arange(seq_len) > shift_point, 3.0, 0),  # ← speed spike
            np.sin(np.pi*t) + rng.normal(0, 0.05, seq_len),
            np.cos(np.pi*t) + rng.normal(0, 0.05, seq_len),
            rng.normal(0, 3, seq_len),   # ← large temp anomaly
            rng.normal(-1, 1, seq_len),  # ← NDVI drop
            t + rng.uniform(0.1, 0.2)   # ← timing shift
        ], axis=1)
        anomalous.append(seq)

    return normal, anomalous



if __name__ == '__main__':
    print("=" * 60)
    print("MigrateIQ — Anomaly Detector — Quickstart")
    print("=" * 60)

    SEQ_LEN = 60
    device  = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. Generate data
    normal_seqs, anomalous_seqs = generate_season_sequences(
        n_normal=200, n_anomalous=40, seq_len=SEQ_LEN
    )
    print(f"Normal seasons: {len(normal_seqs)} | Anomalous: {len(anomalous_seqs)}")

    # 2. Train only on normal
    dataset = SeasonDataset(normal_seqs)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)

    model   = MigrationAutoencoder(input_dim=8, hidden_dim=64,
                                    latent_dim=16, seq_len=SEQ_LEN)
    trainer = AnomalyTrainer(model, device=device)
    trainer.fit(loader, epochs=30)

    # 3. Calibrate threshold on normal data
    threshold = trainer.calibrate_threshold(normal_seqs, percentile=95)

    # 4. Run detection
    detector = AnomalyDetector(model, threshold, device=device)

    print("\n── Normal Season ──")
    result = detector.detect(normal_seqs[0], 'arctic_tern', 2020)
    print(f"Score={result.anomaly_score:.3f} | Anomaly={result.is_anomaly} | "
          f"Severity={result.severity}")

    print("\n── Anomalous Season ──")
    result = detector.detect(anomalous_seqs[0], 'arctic_tern', 2023)
    print(f"Score={result.anomaly_score:.3f} | Anomaly={result.is_anomaly} | "
          f"Severity={result.severity}")
    print(f"Types: {result.anomaly_type}")
    print(f"Story trigger: {result.story_trigger}")

    # 5. Quick precision check
    scores_normal = []
    scores_anom   = []
    for seq in normal_seqs[:50]:
        r = detector.detect(seq, 'arctic_tern', 2020)
        scores_normal.append(r.is_anomaly)
    for seq in anomalous_seqs:
        r = detector.detect(seq, 'arctic_tern', 2023)
        scores_anom.append(r.is_anomaly)

    fpr = sum(scores_normal) / len(scores_normal)
    tpr = sum(scores_anom)   / len(scores_anom)
    print(f"\nFalse positive rate: {fpr:.2%}")
    print(f"True  positive rate: {tpr:.2%}")
