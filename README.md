# Selective Linkage Precision for Released Medical Vision-Language Models

Code for *Selective Linkage Precision for Released Medical Vision-Language Models*.

Selective linkage precision measures how often a released vision-language model links a released chest radiograph to its withheld report, among the queries an attacker selects by the margin between the best and the second-best candidate score, inside the candidate set the published attributes leave.

## What is here

`config/config.yaml` holds every path, cohort, model and analysis setting, and
`config/serde.py` reads it. Each module below takes the path of that file.

| Part of the study | Module |
| --- | --- |
| Paired image-report pools, splits and attribute harmonization | `data_loader/build_paired_pool.py`, `data_loader/cxr_harmonization.py` |
| Report transformations and the language-model rewrite | `data_loader/report_transforms.py`, `data_loader/llm_local.py` |
| Finding labels read from the report text | `data_loader/report_labeler.py` |
| Image and text embeddings for every release | `encoders/extract_embeddings.py`, `encoders/image_text_encoders.py` |
| Candidate ranking, conditioning on published attributes, and the margin | `linkage/retrieval.py`, `linkage/conditioning.py`, `linkage/instrument.py` |
| Adversaries that hold no vision-language model | `linkage/baselines.py` |
| Absent correct reports and the study-level attack | `linkage/openset.py` |
| Candidate collections larger than one test split | `linkage/union_corpus.py` |
| Projection heads, the private and non-private training, and utility | `mitigation/` |
| Whole-pool linkage | `analysis/e1_linkage.py` |
| Conditioning on each attribute and on both conjunctions | `analysis/e2_conditioning.py` |
| Report characteristics against recoverability | `analysis/e3_report_map.py`, `analysis/e11_multivariable.py` |
| Exposure contrasts between recovered and other reports | `analysis/e4_exposure.py` |
| Linkage and content under each report transformation | `analysis/e5_transform_curve.py` |
| Release variants, the matched control, and their paired contrasts | `analysis/e6_model_defense.py` |
| Rankings by Recall@1 against rankings by selective precision | `analysis/e7_metric_disagreement.py` |
| Adversaries without a shared image-text space | `analysis/e8_no_model_adversary.py` |
| Absent reports, perturbed attributes, and the study-level attack | `analysis/e9_openset.py` |
| Model, attributes, and both on one matched query set | `analysis/e10_matched_adversary.py` |
| Larger candidate collections | `analysis/e12_pool_scale.py` |
| Bootstrap, paired contrasts, and the Benjamini-Hochberg correction | `Inference/` |

Each module's entry point takes the path of the configuration as its first argument:
`build_paired_pool`, `build_report_transforms`, `extract_image_embeddings` and
`extract_text_embeddings`, `build_linkage_records`, `build_conditioned_records`,
`build_baseline_records`, `build_openset_records` and `build_openset_study_level`,
`build_union_corpus_records`, `train_heads`, `run_zero_shot` and `run_linear_probe`, and
`run_e1` to `run_e12`. There is no fixed order to run them in: each reads the outputs of
whatever it needs and raises `MissingInput` when those are not on disk. Every one of them
resumes from where it stopped, skips a unit that is already finished and well formed, and
rebuilds whatever was produced under different build parameters.

## Data

The three sources are obtained from their own custodians under their own terms:
MIMIC-CXR and MIMIC-IV from PhysioNet, CheXpert Plus from Stanford AIMI, and
ReXGradient-160K from its data access and use agreement. This repository redistributes
no image, no report and no label, and no model weights. Point `datasets_root` and the
per-cohort keys of `config/config.yaml` at your own copies.

## Credentials

`hf_token` in `config/config.yaml` is a placeholder. Put your own Hugging Face token
there, or leave the placeholder and export `HF_TOKEN`, and the ungated checkpoints load
anonymously. 

## Packages

Python with numpy, pandas, scipy, scikit-learn, statsmodels, torch, torchvision, transformers, open_clip, torchxrayvision, opacus, pyyaml and tqdm.

## Citation

Tayebi Arasteh S, Lotfinia M, Nebelung S, Truhn D. Selective Linkage Precision for
Released Medical Vision-Language Models. *Proceedings of Machine Learning Research*,
Machine Learning for Health (ML4H), 2026.

## License

MIT, see `LICENSE`.
