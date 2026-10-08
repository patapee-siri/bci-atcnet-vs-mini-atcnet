# Contributing

Thank you for your interest in this project. This repository is a research portfolio project
comparing ATCNet and a lightweight variant (Mini-ATCNet) for motor imagery EEG classification.
Feedback, bug reports, and suggestions are welcome.

## Reporting issues

Please open a GitHub issue and include:

- A clear description of the problem or suggestion
- Steps to reproduce, if reporting a bug (command run, subject and config used)
- Your environment (Python version, TensorFlow version, GPU or CPU)

## Suggesting changes

1. Fork the repository and create a branch from `main`.
2. Make your changes, keeping the existing structure (`src/` for modules, `notebooks/` for the
   reference notebook, `results/` for figures).
3. If a change affects reported results, state clearly which numbers changed and why.
4. Open a pull request describing what was changed and the motivation.

## Scope

Contributions most in line with this project include:

- Subject-independent (leave-one-subject-out) evaluation
- Per-subject hyperparameter tuning
- Ensembling across fold-level models
- Regularization studies for the lightweight configuration

## Code style

- Follow the existing naming and module layout in `src/`.
- Keep comments short and focused on why, not what.
- Do not commit datasets (`.gdf`), trained models (`.keras`), or other large artifacts.

## Questions

For questions about the methodology or results, please open an issue.
