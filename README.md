# SP-FFNN: Supervised Feed-Forward Neural Network for Cell Phenotyping

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
