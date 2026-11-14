# KAVACH-JEPA

A JEPA-style world model for network-intrusion forecasting, with a mechanistic-interpretability toolkit. Code only: no UI, no API, no ledger.

Network telemetry (CIC-IDS-2017 flow CSVs or PCAPs) is cut into windows, each window becomes a small graph, and **NetJEPA** (HGT graph encoder + GRU + probabilistic latent predictor, trained with VICReg and a Gaussian NLL) learns to forecast the next state. Two small heads read the embedding: P(infiltration) and a six-class MITRE ATT&CK stage.

## Layout

```text
backend/
  src/
    ingestion/       CSV and PCAP readers
    features/        windowing and feature extraction
    graph/           window -> heterogeneous graph
    models/          NetJEPA, VICReg loss, concept bottleneck
    scoring/         infiltration and stage heads, calibration, novelty
    training/        window cache, train_v2, train_interpretability
    explainability/  SAE, tuned lens, probes, causal tracing, nnsight lab, plots
    benchmark/       adversarial (PGD) robustness, leave-one-attack-out generalisation
  tests/
notebooks/           Colab: GPU training + interpretability suite, nnsight lab
```

## Run

```bash
cd backend
pip install -r requirements.txt
# put the CIC-IDS-2017 MachineLearningCVE CSVs in data/raw/CSV/MachineLearningCVE/
python -m src.training.build_windows_cache
python -m src.training.train_v2 --epochs 20 --batch-size 128 --noise-std 0.1 --sources csv   # uses CUDA if available
python -m src.training.train_interpretability
python -m src.benchmark.robustness --n 30
```

Or use the notebooks (Colab):
1. `notebooks/KAVACH_Interpretability_GPU.ipynb` downloads the data, trains, and runs the interpretability suite.
2. `notebooks/KAVACH_nnsight_Lab.ipynb` runs activation patching, SAE ablation/steering and ~20 visualisations.

The notebooks expect a `kavach_backend.zip` of the `backend/` folder in `MyDrive/kavach/`.

## Known limitations

- Features are z-scored per capture file (including the validation part), so a feature's meaning depends on the capture's composition. Train-only global statistics would be more faithful.
- The calibration split is drawn at random from overlapping training windows, so the threshold and temperature are tuned on near-duplicates of training data.
- The MachineLearningCVE CSVs have no real timestamps; temporal context and forecasts follow file order.
- A window is labelled an attack if any of its 100 rows is; CIC-IDS-2017 also has known mislabelling.
- Unseen attack families are mostly missed, and tiny input perturbations can flip detections.
- CSV graphs have one network node and a placeholder host (no IPs), so node-level analysis needs PCAP windows.
- The nnsight lab has not been validated end to end; run its self-check cell first.

## License

MIT, see [LICENSE](LICENSE).
