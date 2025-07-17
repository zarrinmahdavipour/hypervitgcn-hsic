## Cross-Domain Generalization Experiments
To run cross-domain experiments:
1. Ensure datasets (IndianPines.mat, PaviaU.mat, Houston2013.mat, Botswana.mat) are in the `datasets/` directory.
2. Update `config.yaml` with `use_domain_adaptation: True`, `cross_domain: True`, and `target_datasets: [Houston2013, PaviaU, Botswana]`.
3. Run: `python main.py --config config.yaml`
The script will train on Indian Pines and evaluate zero-shot and fine-tuned performance on target datasets.
