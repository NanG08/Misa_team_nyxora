"""
MigrateIQ — Model 2: Habitat Suitability Scorer
================================================
Scores any geographic grid cell on how suitable it is for a
given species at a given point in time. Used for:
  - Generating the risk zone overlay on the map
  - Identifying corridor bottlenecks
  - Flagging areas where predicted routes pass through unsuitable habitat

Architecture: XGBoost Classifier + SHAP explainability
Input:  (lat, lon, month, temp_c, rainfall_mm, landuse_class,
         elevation_m, ndvi, dist_to_water_km, human_footprint_idx)
Output: suitability score [0.0 – 1.0] + feature importance

Install: pip install xgboost shap scikit-learn numpy pandas
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score,
                              classification_report, confusion_matrix)
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')



FEATURE_COLS = [
    'lat',                  # float  : latitude
    'lon',                  # float  : longitude
    'month',                # int    : 1–12
    'temp_c',               # float  : mean temperature
    'temp_range_c',         # float  : diurnal temperature range
    'rainfall_mm',          # float  : monthly precipitation
    'landuse_code',         # int    : 0=forest,1=grassland,2=wetland,3=crop,4=urban
    'elevation_m',          # float  : elevation
    'slope_deg',            # float  : terrain slope
    'ndvi',                 # float  : vegetation index 0–1
    'dist_water_km',        # float  : distance to nearest water body
    'human_footprint',      # float  : 0–1 human impact index
    'protected_area',       # binary : 1 if inside protected area
    'sst_c',                # float  : sea surface temp (for marine species)
]

LANDUSE_LABELS = {0:'forest', 1:'grassland', 2:'wetland', 3:'cropland', 4:'urban'}


def generate_habitat_data(n_samples: int = 50_000, species: str = 'arctic_tern'):
    """
    Generates synthetic habitat suitability records.
    In production: replace with GBIF occurrence data + WorldClim/MODIS rasters.

    Label = 1 (suitable) if conditions match known species preferences.
    """
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Species-specific preference profiles
    prefs = {
        'arctic_tern':  dict(temp=(5, 25),  rain=(20, 200),  landuse=[0,1,2], elev=(0,500)),
        'monarch':      dict(temp=(15, 35), rain=(30, 150),  landuse=[0,1,3], elev=(0,3000)),
        'wildebeest':   dict(temp=(18, 38), rain=(300, 900), landuse=[1,2],   elev=(0,2000)),
        'humpback':     dict(temp=(-2, 20), rain=(0, 9999),  landuse=[],      elev=(-5000, 0)),
    }
    pref = prefs.get(species, prefs['arctic_tern'])

    data = {
        'lat':              rng.uniform(-70, 80,  n_samples),
        'lon':              rng.uniform(-180, 180, n_samples),
        'month':            rng.integers(1, 13,    n_samples),
        'temp_c':           rng.uniform(-20, 45,   n_samples),
        'temp_range_c':     rng.uniform(2, 25,     n_samples),
        'rainfall_mm':      rng.uniform(0, 500,    n_samples),
        'landuse_code':     rng.integers(0, 5,     n_samples),
        'elevation_m':      rng.uniform(-200, 4000, n_samples),
        'slope_deg':        rng.uniform(0, 60,     n_samples),
        'ndvi':             rng.uniform(0, 1,      n_samples),
        'dist_water_km':    rng.exponential(30,    n_samples),
        'human_footprint':  rng.beta(1.5, 5,       n_samples),
        'protected_area':   rng.integers(0, 2,     n_samples),
        'sst_c':            rng.uniform(-2, 30,    n_samples),
    }
    df = pd.DataFrame(data)

    # Label based on species preferences (with noise for realism)
    t_lo, t_hi = pref['temp']
    r_lo, r_hi = pref['rain']
    temp_ok  = df['temp_c'].between(t_lo, t_hi)
    rain_ok  = df['rainfall_mm'].between(r_lo, r_hi)
    land_ok  = df['landuse_code'].isin(pref['landuse']) if pref['landuse'] else True
    human_ok = df['human_footprint'] < 0.6
    ndvi_ok  = df['ndvi'] > 0.2

    suitability_score = (
        temp_ok.astype(float)  * 0.30 +
        rain_ok.astype(float)  * 0.20 +
        (land_ok if isinstance(land_ok, pd.Series) else pd.Series(land_ok, index=df.index)).astype(float) * 0.20 +
        human_ok.astype(float) * 0.20 +
        ndvi_ok.astype(float)  * 0.10
    )

    # Add noise and threshold at 0.5
    suitability_score += rng.normal(0, 0.08, n_samples)
    df['suitable'] = (suitability_score >= 0.5).astype(int)
    df['species']  = species

    print(f"Generated {n_samples} samples | "
          f"Suitable: {df['suitable'].mean()*100:.1f}%")
    return df



class HabitatSuitabilityModel:
    """
    XGBoost-based habitat suitability classifier.
    Outputs probability [0,1] that a location is suitable for a species.
    """

    def __init__(self, species: str = 'unknown'):
        self.species = species
        self.model   = None
        self.feature_cols = FEATURE_COLS

        self.params = {
            'n_estimators':     500,
            'max_depth':        6,
            'learning_rate':    0.05,
            'subsample':        0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 5,
            'gamma':            0.1,
            'reg_alpha':        0.1,    # L1
            'reg_lambda':       1.0,    # L2
            'scale_pos_weight': 1,      # set to neg/pos ratio if imbalanced
            'random_state':     42,
            'n_jobs':           -1,
            'eval_metric':      'auc',
            'early_stopping_rounds': 30,
        }

    def prepare(self, df: pd.DataFrame):
        X = df[self.feature_cols].copy()
        y = df['suitable'].values
        return X, y

    def train(self, df: pd.DataFrame):
        X, y = self.prepare(df)

        # Handle class imbalance
        pos, neg = y.sum(), (1 - y).sum()
        self.params['scale_pos_weight'] = neg / pos
        print(f"Class ratio neg/pos = {neg/pos:.2f} → scale_pos_weight set")

        # Cross-validated training
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof_probs = np.zeros(len(y))
        fold_aucs = []

        X_arr = X.values

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y)):
            X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
            y_tr, y_val = y[tr_idx],     y[val_idx]

            model = xgb.XGBClassifier(**self.params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            probs = model.predict_proba(X_val)[:, 1]
            oof_probs[val_idx] = probs
            auc = roc_auc_score(y_val, probs)
            fold_aucs.append(auc)
            print(f"  Fold {fold+1}/5 | AUC = {auc:.4f}")

        print(f"\nOOF AUC = {roc_auc_score(y, oof_probs):.4f} "
              f"± {np.std(fold_aucs):.4f}")

        # Final model on all data
        self.model = xgb.XGBClassifier(**self.params)
        X_tr, X_val, y_tr, y_val = train_test_split(X_arr, y, test_size=0.1,
                                                     stratify=y, random_state=42)
        self.model.fit(X_arr, y,
                       eval_set=[(X_val, y_val)],
                       verbose=False)
        print(f"Final model trained on {len(y)} samples.")
        return oof_probs

    def score(self, locations: pd.DataFrame) -> np.ndarray:
        """
        Score a DataFrame of locations.
        Returns suitability probability array [0,1] per row.
        """
        assert self.model is not None, "Train the model first."
        X = locations[self.feature_cols].values
        return self.model.predict_proba(X)[:, 1]

    def score_grid(self, lat_range, lon_range, month: int,
                   climate_fn=None, resolution: int = 50) -> dict:
        """
        Score a full geographic grid for visualisation.

        Args:
            lat_range  : (min_lat, max_lat)
            lon_range  : (min_lon, max_lon)
            month      : 1–12
            climate_fn : callable(lat, lon) → dict of climate vars
            resolution : grid size per side

        Returns:
            dict with 'lats', 'lons', 'scores' (resolution x resolution)
        """
        lats = np.linspace(lat_range[0], lat_range[1], resolution)
        lons = np.linspace(lon_range[0], lon_range[1], resolution)
        grid_lat, grid_lon = np.meshgrid(lats, lons)
        flat_lat = grid_lat.ravel()
        flat_lon = grid_lon.ravel()

        n = len(flat_lat)
        # Default climate values (replace with real raster lookups)
        df = pd.DataFrame({
            'lat':            flat_lat,
            'lon':            flat_lon,
            'month':          np.full(n, month),
            'temp_c':         np.random.uniform(5, 30, n),
            'temp_range_c':   np.random.uniform(5, 20, n),
            'rainfall_mm':    np.random.uniform(20, 300, n),
            'landuse_code':   np.random.randint(0, 5, n),
            'elevation_m':    np.random.uniform(0, 2000, n),
            'slope_deg':      np.random.uniform(0, 30, n),
            'ndvi':           np.random.uniform(0.2, 0.8, n),
            'dist_water_km':  np.random.exponential(20, n),
            'human_footprint':np.random.beta(1.5, 5, n),
            'protected_area': np.random.randint(0, 2, n),
            'sst_c':          np.random.uniform(-2, 25, n),
        })

        scores = self.score(df).reshape(resolution, resolution)
        return {'lats': lats, 'lons': lons, 'scores': scores}

    def explain(self, locations: pd.DataFrame, top_n: int = 5):
        """
        Returns top feature drivers per location using built-in XGBoost importance.
        For full SHAP: install shap and replace with shap.TreeExplainer.
        """
        importances = self.model.feature_importances_
        feat_imp = sorted(zip(self.feature_cols, importances),
                          key=lambda x: x[1], reverse=True)
        print(f"\nTop {top_n} features driving habitat suitability:")
        for feat, imp in feat_imp[:top_n]:
            bar = '█' * int(imp * 100)
            print(f"  {feat:<22} {imp:.4f}  {bar}")
        return dict(feat_imp)

    def evaluate(self, df: pd.DataFrame):
        X, y = self.prepare(df)
        probs = self.model.predict_proba(X.values)[:, 1]
        preds = (probs >= 0.5).astype(int)
        print("\nEvaluation Report:")
        print(classification_report(y, preds,
              target_names=['Unsuitable', 'Suitable']))
        print(f"ROC-AUC: {roc_auc_score(y, probs):.4f}")
        return {'auc': roc_auc_score(y, probs), 'f1': f1_score(y, preds)}

    def save(self, path: str = 'habitat_model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved → {path}")

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)



def analyse_corridor_risk(route_waypoints: list[dict],
                           model: HabitatSuitabilityModel,
                           month: int) -> dict:
    """
    Given a list of route waypoints, scores each point and
    identifies bottlenecks (low-suitability segments).

    Args:
        route_waypoints : list of {'lat':..., 'lon':...}
        model           : trained HabitatSuitabilityModel
        month           : current migration month

    Returns:
        dict with per-point scores and bottleneck indices
    """
    n = len(route_waypoints)
    df = pd.DataFrame(route_waypoints)
    df['month']          = month
    df['temp_c']         = 20.0  # replace with real lookup
    df['temp_range_c']   = 10.0
    df['rainfall_mm']    = 80.0
    df['landuse_code']   = 1
    df['elevation_m']    = 500.0
    df['slope_deg']      = 5.0
    df['ndvi']           = 0.6
    df['dist_water_km']  = 10.0
    df['human_footprint']= 0.2
    df['protected_area'] = 0
    df['sst_c']          = 15.0

    scores       = model.score(df)
    bottlenecks  = np.where(scores < 0.35)[0].tolist()
    risk_level   = 'HIGH' if len(bottlenecks) > n * 0.3 else \
                   'MEDIUM' if len(bottlenecks) > n * 0.1 else 'LOW'

    return {
        'scores':      scores.tolist(),
        'bottlenecks': bottlenecks,
        'risk_level':  risk_level,
        'mean_score':  float(scores.mean()),
        'min_score':   float(scores.min()),
    }


# ─────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("MigrateIQ — Habitat Suitability Model — Quickstart")
    print("=" * 60)

    species = 'monarch'
    df = generate_habitat_data(n_samples=30_000, species=species)

    model = HabitatSuitabilityModel(species=species)
    model.train(df)
    model.evaluate(df.sample(5000, random_state=99))
    model.explain(df.head(1))

    # Score a corridor
    route = [{'lat': 30+i, 'lon': -100} for i in range(20)]
    risk = analyse_corridor_risk(route, model, month=9)
    print(f"\nCorridor Risk: {risk['risk_level']}")
    print(f"Mean suitability: {risk['mean_score']:.3f}")
    print(f"Bottleneck points: {risk['bottlenecks']}")

    model.save('habitat_monarch.pkl')