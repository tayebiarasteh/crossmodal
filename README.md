# Selective Linkage Precision for Released Medical Vision-Language Models

A released medical vision-language model embeds chest radiographs and radiology reports in one space. So anyone who downloads it can score one radiograph against a corpus of reports and rank them. This repository measures how often the top-ranked report is the right one at the operating point that an adversary chooses, instead of averaged over a whole dataset. It also measures four release variants, meaning one differentially private defense, its comparator, an adversary's bypass, and a control that destroys the alignment without any training.

## Repository layout

| Directory | Contents |
| --- | --- |
| `config/` | The single configuration file and its loader. One file sets every path, dataset, model, report form, and threshold |
| `data_loader/` | Dataset pool construction, report section parsing, the eight report forms, and the client for the locally served language model |
| `encoders/` | Image and text embedding extraction and caching, one cache per model, dataset, and report form |
| `linkage/` | Ranking against the candidate corpus, the measurements computed from it, the conditioned candidate pools, and the two adversaries that use no vision-language model |
| `mitigation/` | The four release variants and the clinical performance measured under each |
| `analysis/` | One module per experiment, each emitting canonical result rows |
| `Inference/` | Bootstrap intervals over patient clusters, tests, multiple-comparison correction, and the row schema that every reported number is emitted under |
| `generation/` | Wrappers for the served open-weight language model |

## Before running

```bash
git clone https://github.com/tayebiarasteh/crossmodal.git
```

Every absolute path in `config/config.yaml` is written as `/PATH` and has to be replaced with the location on your own machine. The four machine roots at the top of that file and `GLOBAL_CONFIG_PATH` at the top of `main_crossmodal.py` are the only per-machine lines. Everything else is built from them.

The three datasets are not redistributed here. MIMIC-CXR and CheXpert Plus are obtained from PhysioNet under their own data use agreements, and ReXGradient-160K from its own release. The PhysioNet agreement forbids sending report text to a third-party service. So the language model behind the report rewriting and the label extraction is served locally, and no provider model is reachable from the code.

Two secrets stay as placeholders in the configuration and are read from the environment instead. `CROSSMODALITY_API_BASE_URL` and `CROSSMODALITY_API_KEY` point at the OpenAI-compatible server that serves the open-weight language model. The `hf_token` key is an access token for the gated checkpoint in the model panel. Leaving it as the placeholder loads every other checkpoint anonymously.

## Running the pipeline

Each pipeline step is a `main_*` function in `main_crossmodal.py`. A step takes the path to `config/config.yaml` and returns when it is complete.

```bash
python3 -c "import main_crossmodal as m; m.main_audit_wiring()"
python3 -c "import main_crossmodal as m; m.main_build_paired_pool(cohort='mimic')"
```

The dependency order is one paired pool per dataset, then the report forms, then the cached image and text embeddings for each model, then the ranking records, and finally the analyses and the result tables. The mitigation branch trains the projection heads, re-embeds the cached features through them, and measures clinical performance before its own analysis runs. `main_audit_wiring` checks the wiring before any of that costs time. It reports a configuration key that nothing reads, a declared name that nothing implements, a missing module, and an embedding width that disagrees with the checkpoint's own config.

Every step is resumable. It writes per unit of work as it goes and skips a unit that is already finished and well-formed. So re-running a killed step continues it instead of starting over. Each cache and checkpoint also stores the parameters that it was built under, and a resume compares them before it applies a byte. A changed model, resolution, or dataset manifest therefore forces a named rebuild instead of reusing an out-of-date artifact.

The analysis layer is CPU-bound and re-runs in minutes once the embedding caches exist. Building those caches and training the projection heads are the two parts that need a GPU.

Every number we report is emitted as a row carrying its metric, its bootstrap interval, its resampling unit, and the keys that it was computed under. Any figure in the paper can therefore be traced back to the module that produced it.

## Citation

```bibtex
@article{TayebiArasteh2026SelectiveLinkage,
  title   = {Selective Linkage Precision for Released Medical Vision-Language Models},
  author  = {Tayebi Arasteh, Soroosh and Lotfinia, Mahshad and Nebelung, Sven and Truhn, Daniel},
  journal = {arXiv preprint arXiv:2606.02276},
  year    = {2026}
}
```

