\# MISA - AI/ML Model Documentation



---





\*\*An AI-powered platform that tracks, predicts, and explains animal migration patterns for the general public.\*\*



---



\## What It Does



Misa uses four ML models working as a single pipeline — from raw GPS tracking data to public-facing story cards explaining \*why\* migrations are changing.



```

GPS + Climate Data → Route Prediction → Habitat Scoring → Anomaly Detection → Story Generation

```



---



\## The 4 Models



\### 1. Route Predictor `route\_predictor.py`

\*\*Encoder-Decoder LSTM with Bahdanau Attention\*\*



Takes 30 timesteps of historical GPS + climate data (temperature, wind, sea surface temp, NDVI) and predicts the next set of future migration waypoints. Uses Monte-Carlo Dropout to generate uncertainty confidence zones shown on the map.



\### 2. Habitat Suitability Scorer `habitat\_scorer.py`

\*\*XGBoost Classifier\*\*



Scores any geographic grid cell from 0–1 on how suitable it is for a given species at a given time. Trained on 14 features including land use, elevation, NDVI, rainfall, and human footprint index. Powers the risk zone overlays and corridor bottleneck detection on the map.



\### 3. Anomaly Detector `anomaly\_detector.py`

\*\*LSTM Autoencoder\*\*



Trained only on \*normal\* migration seasons. When a new season produces high reconstruction error, it's flagged as anomalous. Detects route deviations, timing shifts, speed irregularities, and habitat degradation signals. Triggers the story generation pipeline when fired.



\### 4. Story Event Linker `story\_linker.py`

\*\*RAG + LLM API\*\*



Takes an anomaly signal, semantically searches a FAISS vector index of real-world environmental events (droughts, wildfires, deforestation, ocean temperature anomalies), and passes the most relevant events to Claude to generate the public-facing cause-and-effect story cards shown in the \*"Why It's Changing"\* feed.



---



\## Tech Stack



| Layer | Tools |

|---|---|

| Deep Learning | PyTorch (LSTM, Autoencoder) |

| Gradient Boosting | XGBoost |

| Vector Search | FAISS + sentence-transformers |

| LLM / Story Gen | Anthropic Claude API (RAG) |

| Data | Movebank, GBIF, NASA Earthdata, WorldClim |

| API Server | FastAPI |



---



\## AMD Integration



All PyTorch models run on \*\*AMD Instinct MI300X/MI325X\*\* via the ROCm stack with zero code changes. XGBoost runs on \*\*AMD EPYC 9004\*\* CPUs. The LLM layer uses \*\*AMD Instinct MI355X\*\*. Switch from CUDA to ROCm in one line:



```bash

pip install torch --index-url https://download.pytorch.org/whl/rocm6.0

```



---



\## Quickstart



```bash

pip install -r requirements.txt

export LLM\_API\_KEY=your\_key         # for story generation

python pipeline.py                  # runs all 4 models end-to-end

```

