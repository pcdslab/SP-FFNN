# SP-FFNN: Supervised Feed-Forward Neural Network for Cell Phenotyping

# Abstract
Accurate cell-type annotation from high-plex spatial proteomics images is a fundamental and labor-intensive step in tissue biology. Feed-Forward Neural Network (FFNN) trained directly on single-cell marker-expression vectors can match pathologist-level annotation accuracy while remaining far more computationally efficient than deep image-based classifiers. However, it remains unclear how much of this advantage stems specifically from the use of a neural network rather than from the broader supervised, expression-matrix-based learning paradigm shared by simpler classical algorithms. In this work, we benchmark a four-hidden-layer FFNN against four classical machine learning baselines of Logistic Regression (LR), Support Vector Machine (SVM), Random Forest (RF), and Decision Tree (DT) on the publicly available datasets i.e., colorectal cancer (CRC) CODEX, highly multiplexed MIBI from classical Hodgkin lymphoma (cHL), diffuse large B cell lymphoma (DLBCL), and an in house cHL CODEX. The SP-FFNN achieved the highest overall accuracy (97.63%) and weighed F1-score (87.88%), outperforming Random Forest (accuracy 84.94%, weighted F1 84.73%), the next-best classical method, by a substantial margin. Our results provide a meaningful accuracy advantage over both linear and tree-based classical models for high-dimensional, multiplexed proteomic cell phenotyping, while remaining inexpensive enough to train on commodity hardware.

© This code is made available for non-commercial academic purposes.

This repository contains the implementation of a supervised feed-forward neural network (SP-FFNN) and classical machine learning baselines for cell phenotyping using spatial proteomics data.

---

## Repository Structure
```
.
├── data/
│   └── cell_phenotyping/
│       ├── train.csv
│       └── valid.csv
├── FFNN.py                # Trains the feed-forward neural network
├── ML-Methods.py          # Trains classical ML baselines (LogReg, SVM, RF, ExtraTrees)
├── EvaluationFFNN.py       # Generates confusion matrix, loss/accuracy plots
├── requirements.txt
└── README.md
```

---

## Installation

### Pre-requisites
* Python (3.9.0)
* PyTorch (1.13.1)
* NumPy (1.23.5)
* Pandas (1.5.5)
* Scikit-learn (1.2.1)
* Matplotlib
* Seaborn


### Virtual Environment (Optional but recommended)

**Using Anaconda**
```shell
conda create -n FFNN python=3.9.0
conda activate FFNN
```

**Using Python venv**
```shell
# for Unix/macOS
python3 -m pip install --user virtualenv
python3 -m venv FFNN
source FFNN/bin/activate

# for Windows
py -m venv FFNN
.\FFNN\Scripts\activate
```

### Install Dependencies
Once your environment is activated, install the required packages:
```shell
pip install -r requirements.txt
```

---

## Datasets

Datasets are expected to be prepared in CSV format containing at least **N+1** columns, where **N** is the number of markers in the given dataset, plus one additional column for cell size. Each row of the CSV file should represent marker expression for a given cell along with cell size.

To retrain cell phenotyping on a new dataset:
1. Split the dataset into `train.csv` and `valid.csv` files.
2. Add an extra column `cell_label` (numerical, e.g. `0, 1, 2, ...` — not strings such as `'CD4 T cell'`, `'CD8 T cell'`) to both files to provide the ground-truth class ID for each cell.

Sample train/valid CSV files are provided under `data/cell_phenotyping/`.

---

## Training

Before running, update the file paths at the top of `FFNN.py` and `ML-Methods.py` to point to your data:
```python
TRAIN_CSV = '../data/cell_phenotyping/train.csv'
VALID_CSV = '../data/cell_phenotyping/valid.csv'
```

Then run:
```shell
python FFNN.py
python ML-Methods.py
```

---

## Results / Evaluation

After training, generate the confusion matrix and training/validation loss and accuracy plots:
```shell
python EvaluationFFNN.py
```
Output plots and metrics are saved to `results_ffnn/ and results_classical/`.

---


## Contact
For questions, please contact zubairsaeed602@gmail.com or open an issue in this repository.
