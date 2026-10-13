# PathoTraj

**Time-aligned multimodal trajectory modeling for rolling prediction of organ-specific deterioration in hospitalized adults with cardiorenal-metabolic conditions**

This repository accompanies the manuscript:

> Han Z (Independent Researcher). *PathoTraj: Time-Aligned Multimodal Trajectory Modeling for Rolling Prediction of Organ-Specific Deterioration in Hospitalized Adults with Cardiorenal-Metabolic Conditions.* Manuscript under review.

> **Status.** The manuscript is under peer review. This repository currently documents the study design and the analysis pipeline. The complete code for cohort construction, model training, and evaluation will be released here, as a streamlined package, after the manuscript is accepted. No patient data are, or will be, stored in this repository.

---

## What PathoTraj does

PathoTraj gives hospitalized adults with prior cardiorenal-metabolic conditions a rolling, organ-specific risk estimate. Starting 24 h after admission and then every 12 h, it estimates the risk of four outcomes within the next 24 and 48 h:

| Outcome | Operational definition (summary) |
|---|---|
| Renal decompensation | KDIGO creatinine criteria (rise ≥0.3 mg/dL within 48 h or ≥1.5× within 168 h) or new renal replacement therapy; prior ESRD excluded |
| Pulmonary decompensation | ≥2 SpO₂ measurements <88% within 6 h, or new respiratory support |
| Hemodynamic decompensation | ≥2 MAP measurements <65 mmHg within 4 h, or new vasoactive support |
| Life-support escalation | New ICU transfer or ICU-stay start, vasoactive or respiratory support, or renal replacement therapy |

At each prediction time (landmark), the model combines four input branches. Every input is restricted to information that was available in the record at that time:

- **Structured events**: up to 256 laboratory, bedside, and treatment events from the previous 120 h. The events are selected for their relevance to organ decompensation and grouped into 23 physiological pathways.
- **Radiology reports**: up to 8 reports whose chart time *and* storage time precede the landmark. Each report is embedded once with a frozen sentence encoder.
- **Static context**: 31 features.
- **Aggregate summaries**: 122 features.

A Fusion Transformer combines the four branches. Four outcome-specific heads output discrete-time interval hazards (0–6, 6–12, 12–24, and 24–48 h), and these are combined into cumulative 24- and 48-h risks.

## Data access

The study uses **MIMIC-IV v3.1** and **MIMIC-IV-Note v2.2**. These databases are available only to credentialed users through PhysioNet:

- MIMIC-IV: https://physionet.org/content/mimiciv/
- MIMIC-IV-Note: https://physionet.org/content/mimic-iv-note/

To use the released code, you will need to:

1. Complete PhysioNet credentialing and the required human-subjects training.
2. Sign the data use agreements for both databases.
3. Download the data to your own environment.

Under the data use agreement, this repository will never contain raw data, derived patient-level datasets, embeddings, predictions, or model weights trained on MIMIC.

## Study design at a glance

| Item | Setting |
|---|---|
| Cohort | 49,958 patients; 159,071 admissions; 1,751,439 landmarks |
| Partitions | Patient-disjoint, by `anchor_year_group`: training 2008–2013, validation 2014–2016, test 2017–2019 |
| Landmarks | 24 h after admission, then every 12 h until discharge; look-back ≤120 h |
| Horizons | 24 h (primary) and 48 h |
| Primary metric | Admission-weighted macro AUPRC |
| Comparators | Static + aggregate LightGBM, neural static + aggregate model, Event-GRU, Event-only Transformers (pretrained / scratch) |
| Experiment matrix | 25 fixed configurations × 3 seeds (17, 42, 73) = 75 supervised runs, plus 15 pretraining runs |
| Uncertainty | Paired patient-level bootstrap (500 resamples per seed; Benjamini–Hochberg within the prespecified 60-test family); 95% CIs of the three-seed mean from 1,000 joint patient-level resamples |

## Pipeline

The released code will follow these stages:

| Stage | Purpose | Main components |
|---|---|---|
| 1. Preflight | Check inputs, package identity, and available resources | `scripts/launch_all.py`, `src/run_revision.py --stage preflight` |
| 2. Build | Cohort, landmarks, outcome labels and masks; source- and availability-time cleaning of events; bounded event links; aggregates; radiology note links; frozen report embeddings | `src/build_cardiorenal_pulmonary_trajectory_tokens.py`, `src/rebuild_source_events.py`, `src/bounded_event_links.py`, `src/build_radiology_embeddings_v4_1.py`, `src/build_train_ready_dataset_v4_1.py` |
| 3. Pretrain | Self-supervised event-encoder pretraining (event–report and event-only variants) | `src/pretrain_event_transformer_v4_1.py`, `src/pretrain_event_only.py` |
| 4. Train | Supervised training of the 25 configurations × 3 seeds, including LightGBM and GRU comparators | `src/train_multimodal_transformer_v4_1.py`, `src/train_v4_1_lightgbm_baseline.py`, `src/model_v4_1.py` |
| 5. Analyze | Discrimination, calibration (Platt and isotonic), paired bootstrap, onset-distance sensitivity, subgroups, alert operating points, decision curves, occlusion | `src/revision_analysis.py`, `src/extended_analysis.py`, `src/analysis_metrics_v4_1.py` |
| 6. Audit | Temporal and linkage audit; report-pathway leakage audit; MIMIC-naive encoder substitution; bootstrap CIs | `src/storage_audits.py`, `audit/audit_text_pathway.py`, `audit/text_free_bound.py`, `audit/encoder_swap/`, `audit/ci_bootstrap.py` |
| 7. Export | Aggregate tables and figures only; no patient-level data | `src/run_revision.py --stage export` |

Planned usage (the interface may be simplified in the release):

```bash
pip install -r requirements.txt
python scripts/launch_all.py \
  --raw-root /path/to/mimiciv/3.1 \
  --radiology-path /path/to/mimic-iv-note/2.2/note/radiology.csv.gz \
  --work-dir /path/to/workdir \
  --stage all --profile full --device cuda
```

A synthetic smoke test that uses no MIMIC data will be included (`scripts/run_synthetic_smoke.py`). Its outputs are permanently marked as synthetic and cannot be mistaken for study results.

## Safeguards against information leakage

- **Two times per input.** An event or report is used only if both its source time and the time it became available in the record precede the landmark.
- **No future admission information.** Current-admission diagnoses, final length of stay, and time to discharge are never predictors.
- **Training-only fitting.** Vocabularies, numerical transforms, aggregates, and linking maps are fitted on the training partition only.
- **Fixed experiment matrix.** Checkpoints, thresholds, and calibrators are selected on validation data, and test labels are used only for evaluation.
- **Report-pathway audit.** The public report encoder's base model was pretrained on a corpus that included MIMIC-IV-Note. The audit therefore checks code paths and run configurations, cross-partition duplicate reports, and models that never use report text, and it re-runs the Full model with a MIMIC-naive encoder (`NeuML/pubmedbert-base-embeddings`).

## Main settings

| Parameter | Value |
|---|---|
| Model dimension / attention heads | 192 / 6 |
| Event / radiology / fusion layers | 3 / 1 / 1 |
| Max events / reports per landmark | 256 / 8 |
| Dropout | 0.15 |
| Optimizer | AdamW; learning rate 3×10⁻⁵; weight decay 1×10⁻⁴ |
| Batch size / max epochs | 128 / 20; early stopping on validation macro AUPRC (patience 5) |
| Pretraining | 5 epochs; mask rate 0.15; learning rate 1×10⁻⁴ |
| Positive-class weight cap | 50 |
| Report encoder | `NeuML/bioclinical-modernbert-base-embeddings` (frozen; 1,024 tokens; mean-pooled; L2-normalized) |
| LightGBM | Up to 500 estimators with validation early stopping |

## Requirements

- Python ≥ 3.10
- CUDA GPU recommended for training. CPU is sufficient for the synthetic smoke test.
- Core packages: `numpy`, `pandas`, `pyarrow`, `duckdb`, `scikit-learn`, `torch`, `lightgbm`, `tqdm`, and `sentence-transformers` (for report embeddings).

## Repository layout (planned)

```
Pathotraj/
├── README.md
├── requirements.txt
├── configs/        # dataset definition, comparison registry
├── src/            # cohort/event build, models, training, analysis
├── scripts/        # launcher, synthetic smoke test
├── audit/          # leakage audits, encoder substitution, bootstrap CIs
└── tests/          # unit tests (synthetic data only)
```

## Citation

If you use this work, please cite the manuscript. The citation will be updated after publication.

```bibtex
@article{han_pathotraj,
  author  = {Han, Ziling},
  title   = {PathoTraj: Time-Aligned Multimodal Trajectory Modeling for Rolling Prediction of Organ-Specific Deterioration in Hospitalized Adults with Cardiorenal-Metabolic Conditions},
  journal = {Under review},
  year    = {2026}
}
```

Please also cite MIMIC-IV, MIMIC-IV-Note, and PhysioNet as required by their data use agreements.

## License

The license will be added with the code release.

## Contact

Ziling Han (Independent Researcher) — bingqilingzzzz@gmail.com

Repository: https://github.com/BINGQILINGYA/Pathotraj
