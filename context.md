# Project Context

## Overview
This repository is organized around two separate workflows:

1. Dataset generation module
2. Closed-loop module

The modules are intentionally isolated so development can continue independently.

## Current Structure

- data/
  - datasets/
  - logs/
- models/
- plots/
- scripts/
  - setup_namespaces.sh
  - dataset_generation/
    - dynamic_traffic_generator.py
    - generate_dataset.py
    - health_sender.py
    - health_receiver.py
    - run_experiment.sh
    - train_model.py
    - tune_xgboost.py
    - plot.py
    - bd.py
  - closed_loop/
    - health_sender.py
    - health_receiver.py
    - run_closed_loop_stress_auto.sh

## Module Boundaries

### Dataset generation
- Uses scripts inside scripts/dataset_generation only.
- Has its own local copies of sender and receiver.
- Writes telemetry and generated datasets under data/.
- Intended for synthetic data creation, feature engineering, and model training input preparation.

### Closed-loop
- Uses scripts inside scripts/closed_loop only.
- Operates as a separate runtime/stress workflow.
- Does not depend on dataset_generation sender or receiver copies.

## Entry Points

### Dataset generation flow
1. scripts/setup_namespaces.sh
2. scripts/dataset_generation/run_experiment.sh
3. scripts/dataset_generation/dynamic_traffic_generator.py
4. scripts/dataset_generation/generate_dataset.py
5. scripts/dataset_generation/train_model.py
6. scripts/dataset_generation/tune_xgboost.py

### Closed-loop flow
1. scripts/setup_namespaces.sh
2. scripts/closed_loop/run_closed_loop_stress_auto.sh

## Notes
- Namespace setup remains shared via scripts/setup_namespaces.sh.
- Sender and receiver are duplicated by design across both modules.
- Keep all new dataset-generation logic inside scripts/dataset_generation.
- Keep all new closed-loop logic inside scripts/closed_loop.
