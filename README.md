# ARC: Atom–Reactivity Correspondence

This repository contains the implementation of **ARC**, a unified multi-task graph neural network for joint **atom mapping** and **reaction center identification**. This repository accompanies our **NeurIPS 2025 submission**.

## Model Overview

**ARC (Atom-Reactivity Correspondence)** is a unified graph neural network that jointly performs **atom mapping** and **reaction center identification**. These tasks are commonly treated in isolation, but ARC integrates them within a shared representation space.

Key features:
- **Multi-Task Learning**: Combines atom mapping and reaction center identification into a unified multi-task framework.
- **Cross-Graph Attention**: Aligns product atoms with their mapped reactant counterparts to guide attention toward reactive regions.
- **Dual-Graph Representation**: Treats bonds as nodes, enabling localized bond-centric message passing.

ARC achieves **state-of-the-art performance** on the USPTO-50K benchmark. The full training pipeline, and evaluation scripts are included.

---

## Directory Structure

```bash
.
├── arc/                         # Source code
│   ├── model.py                # ARC model
│   ├── training.py             # Training loop
│   ├── evaluation.py           # Evaluation and metrics
│   ├── dataset.py              # Dataset loading and preprocessing
│   ├── utils.py                # Utility functions
│   ├── utils_data.py           # Graph utilities
│   ├── pairdata.py             # Product-reactant pair data
│   
│
├── notebooks/                  # Notebooks
│   ├── test_example.ipynb      # Inference example
│   └── dataset_stats.ipynb     # Dataset analysis
│
│
├── pretrained/                 # Pretrained model weights
│   └── best_model.pt           # Model checkpoint
│
├── run.sh                      # SLURM training script
├── model_utest.py              # Unit tests
├── environment.yml             # Conda environment spec
├── requirements.txt            # Python package dependencies
├── LICENSE
└── README.md
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate arc
```

Or use pip:
```bash
pip install -r requirements.txt
```

---

## Usage

### Train ARC (full model)
```bash
bash run.sh
```

### Evaluate on test set
```bash
python arc/evaluation.py
```

### Inference Example (Jupyter)
```bash
jupyter notebook notebooks/test_example.ipynb
```

---

## Pretrained Weights
Pretrained ARC model weights are available at:
[Download ARC Weights](https://drive.google.com/drive/folders/1UrGDbtgEzqXsuq27rhF7rc8pqa5TL-Cn?usp=sharing)

To load:
```python
model.load_state_dict(torch.load("pretrained/best_model.pt"))
```

---

## Dataset

We use the USPTO-50K dataset, specifically the **canonicalized version** by [Somnath et al. (2021)](https://github.com/somnathrakshit/graphretro), which removes atom-index bias. Labels are derived from atom-mapped reactions:
- Atoms are reactive if formal charge or hydrogen count differs.
- Bonds are reactive if formed, broken, or changed in order.

See `dataset.py` for preprocessing details, and `notebooks/dataset_stats.ipynb` for statistics.


---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
