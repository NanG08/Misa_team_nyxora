"""
MigrateIQ — Pipeline Orchestrator
===================================
Ties all 4 models into a single inference pipeline.
This is what your FastAPI server calls.

Flow:
  Raw GPS + climate data
       ↓
  Model 1 (LSTM)        → predicted route waypoints
       ↓
  Model 2 (XGBoost)     → habitat suitability scores along route
       ↓
  Model 3 (Autoencoder) → anomaly detection on current season
       ↓
  Model 4 (RAG + LLM)   → story card generation if anomaly found
       ↓
  JSON response → frontend

Install: pip install torch xgboost scikit-learn numpy pandas
         pip install faiss-cpu sentence-transformers anthropic
"""

import os
import json
import numpy as np
import pandas as pd
from dataclasses import asdict
from typing import Optional


# ─────────────────────────────────────────────
# IMPORT ALL MODELS
# ─────────────────────────────────────────────

# Model 1
from model1_route_predictor import (
    MigrationLSTM, MigrationDataset, generate_synthetic_data, predict_route
)

# Model 2
from model2_habitat_scorer import (
    HabitatSuitabilityModel, analyse_corridor_risk, generate_habitat_data
)

# Model 3
from model3_anomaly_detector import (
    MigrationAutoencoder, AnomalyDetector, AnomalyTrainer,
    SeasonDataset, generate_season_sequences
)

# Model 4
from model4_story_linker import (
    MigrationStoryPipeline, SAMPLE_EVENTS
)

import torch
from torch.utils.data import DataLoader


# ─────────────────────────────────────────────
# ORCHESTRATOR
# ─────────────────────────────────────────────

class MigrateIQPipeline:
    """
    Single interface to all 4 ML models.
    In production: models are loaded from saved .pt/.pkl files.
    Here: trains from scratch on synthetic data for demo.
    """

    def __init__(self, device: str = 'cpu', api_key: Optional[str] = None):
        self.device = device
        self.api_key = api_key

        self.route_model    = None
        self.habitat_model  = None
        self.anomaly_model  = None
        self.anomaly_thresh = None
        self.story_pipeline = None

        print("MigrateIQ Pipeline initialised.")

    # ── Training ───────────────────────────────

    def train_all(self, quick: bool = True):
        """Train all models. quick=True uses minimal epochs for demo."""
        epochs = 5 if quick else 50
        print("\n" + "="*60)
        print("STEP 1/4 — Training Route Predictor (LSTM)")
        print("="*60)
        self._train_route_model(epochs=epochs)

        print("\n" + "="*60)
        print("STEP 2/4 — Training Habitat Scorer (XGBoost)")
        print("="*60)
        self._train_habitat_model()

        print("\n" + "="*60)
        print("STEP 3/4 — Training Anomaly Detector (Autoencoder)")
        print("="*60)
        self._train_anomaly_model(epochs=epochs)

        print("\n" + "="*60)
        print("STEP 4/4 — Building Story Event Index (RAG)")
        print("="*60)
        self._build_story_pipeline()

        print("\n✓ All models ready.")

    def _train_route_model(self, epochs):
        df       = generate_synthetic_data(n_animals=20, timesteps=200)
        dataset  = MigrationDataset(df, seq_len=30, pred_len=10)
        n_train  = int(0.8 * len(dataset))
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, len(dataset) - n_train]
        )
        train_l  = DataLoader(train_ds, batch_size=32, shuffle=True)
        val_l    = DataLoader(val_ds,   batch_size=32, shuffle=False)

        from model1_route_predictor import RouteTrainer
        self.route_model = MigrationLSTM(input_dim=7, hidden_dim=128,
                                          num_layers=2, dropout=0.2)
        trainer = RouteTrainer(self.route_model, device=self.device)
        trainer.fit(train_l, val_l, epochs=epochs)
        self.route_dataset = dataset  # keep for inference demo

    def _train_habitat_model(self):
        df = generate_habitat_data(n_samples=15_000, species='arctic_tern')
        self.habitat_model = HabitatSuitabilityModel(species='arctic_tern')
        self.habitat_model.train(df)

    def _train_anomaly_model(self, epochs):
        normal_seqs, _ = generate_season_sequences(n_normal=150, n_anomalous=0, seq_len=60)
        dataset = SeasonDataset(normal_seqs)
        loader  = DataLoader(dataset, batch_size=16, shuffle=True)

        self.anomaly_model_nn = MigrationAutoencoder(
            input_dim=8, hidden_dim=64, latent_dim=16, seq_len=60
        )
        trainer = AnomalyTrainer(self.anomaly_model_nn, device=self.device)
        trainer.fit(loader, epochs=epochs)
        self.anomaly_thresh   = trainer.calibrate_threshold(normal_seqs)
        self.anomaly_model    = AnomalyDetector(
            self.anomaly_model_nn, self.anomaly_thresh, device=self.device
        )
        self._normal_seqs = normal_seqs   # keep for demo

    def _build_story_pipeline(self):
        self.story_pipeline = MigrationStoryPipeline(api_key=self.api_key)
        self.story_pipeline.build_index(SAMPLE_EVENTS)

    # ── Inference ──────────────────────────────

    def run(self, species: str, season_df: pd.DataFrame,
            input_seq: np.ndarray, month: int = 6) -> dict:
        """
        Full pipeline inference for one species / one season.

        Args:
            species    : species identifier string
            season_df  : DataFrame with this season's GPS + climate data
            input_seq  : (seq_len, features) array for route prediction
            month      : current month for habitat scoring

        Returns:
            JSON-serialisable dict with all model outputs
        """
        results = {'species': species}

        # ── Model 1: Route Prediction ─────────
        print(f"\n[1/4] Predicting route for {species}...")
        route_pred = predict_route(
            self.route_model, input_seq,
            pred_len=10, n_samples=30, device=self.device
        )
        results['predicted_route'] = {
            'mean':      route_pred['mean'].tolist(),
            'lower_95':  route_pred['lower_95'].tolist(),
            'upper_95':  route_pred['upper_95'].tolist(),
        }
        print(f"   Predicted {len(route_pred['mean'])} waypoints.")

        # ── Model 2: Habitat Scoring ──────────
        print(f"[2/4] Scoring habitat suitability...")
        waypoints = [
            {'lat': float(pt[0]), 'lon': float(pt[1])}
            for pt in route_pred['mean']
        ]
        corridor_risk = analyse_corridor_risk(
            waypoints, self.habitat_model, month=month
        )
        results['habitat'] = corridor_risk
        print(f"   Corridor risk: {corridor_risk['risk_level']} "
              f"(mean suitability: {corridor_risk['mean_score']:.3f})")

        # ── Model 3: Anomaly Detection ────────
        print(f"[3/4] Running anomaly detection...")
        # Use a random normal seq as placeholder (replace with real extract_season_features)
        test_seq  = self._normal_seqs[0]
        anomaly   = self.anomaly_model.detect(test_seq, species, 2024, season_df)
        results['anomaly'] = {
            'score':        anomaly.anomaly_score,
            'is_anomaly':   anomaly.is_anomaly,
            'types':        anomaly.anomaly_type,
            'severity':     anomaly.severity,
            'timing_shift': anomaly.timing_shift_days,
            'route_dev_km': anomaly.route_deviation_km,
        }
        print(f"   Anomaly score={anomaly.anomaly_score:.3f}, "
              f"severity={anomaly.severity}")

        # ── Model 4: Story Card ───────────────
        story_card = None
        if anomaly.is_anomaly:
            print(f"[4/4] Generating story card...")
            card = self.story_pipeline.generate_story(anomaly)
            story_card = {
                'card_id':      card.card_id,
                'title':        card.title,
                'body':         card.body,
                'cause':        card.cause,
                'effect':       card.effect,
                'impact_level': card.impact_level,
                'confidence':   card.confidence,
                'icon':         card.icon,
                'source_events':card.source_events,
            }
            print(f"   Story: {card.title}")
        else:
            print(f"[4/4] No anomaly — no story card generated.")

        results['story_card'] = story_card
        return results


# ─────────────────────────────────────────────
# FASTAPI INTEGRATION EXAMPLE
# ─────────────────────────────────────────────

FASTAPI_EXAMPLE = '''
# Save as server.py and run: uvicorn server:app --reload
# pip install fastapi uvicorn

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from pipeline import MigrateIQPipeline
import pandas as pd

app      = FastAPI(title="MigrateIQ API")
pipeline = MigrateIQPipeline()
pipeline.train_all(quick=True)   # or load from saved weights

class PredictRequest(BaseModel):
    species:   str
    input_seq: list[list[float]]   # (seq_len, features)
    month:     int = 6

@app.post("/predict")
def predict(req: PredictRequest):
    seq = np.array(req.input_seq)
    dummy_df = pd.DataFrame()   # replace with real season data
    result = pipeline.run(
        species    = req.species,
        season_df  = dummy_df,
        input_seq  = seq,
        month      = req.month
    )
    return result

@app.get("/health")
def health():
    return {"status": "ok", "models": ["route", "habitat", "anomaly", "story"]}
'''


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("MigrateIQ — Full Pipeline Demo")
    print("=" * 60)

    pipeline = MigrateIQPipeline(
        device  = 'cuda' if torch.cuda.is_available() else 'cpu',
        api_key = os.environ.get('ANTHROPIC_API_KEY')
    )

    # Train all models
    pipeline.train_all(quick=True)

    # Run inference
    sample_input = pipeline.route_dataset[0][0].numpy()   # (seq_len, features)
    sample_df    = pd.DataFrame({'lat':[50,51,52], 'lon':[-30,-28,-26],
                                 'day_of_year':[90,100,110], 'year':[2024,2024,2024],
                                 'species':['arctic_tern']*3, 'temp_c':[10,11,12],
                                 'ndvi':[0.6,0.65,0.6]})

    output = pipeline.run(
        species   = 'arctic_tern',
        season_df = sample_df,
        input_seq = sample_input,
        month     = 5
    )

    print("\n" + "="*60)
    print("PIPELINE OUTPUT")
    print("="*60)
    print(json.dumps({
        'predicted_route_pts': len(output['predicted_route']['mean']),
        'corridor_risk':       output['habitat']['risk_level'],
        'anomaly_score':       round(output['anomaly']['score'], 3),
        'anomaly_severity':    output['anomaly']['severity'],
        'story_generated':     output['story_card'] is not None,
        'story_title':         output['story_card']['title'] if output['story_card'] else None,
    }, indent=2))

    print("\nFastAPI integration example saved to FASTAPI_EXAMPLE string.")
    print("Set ANTHROPIC_API_KEY env variable to enable LLM story generation.")
