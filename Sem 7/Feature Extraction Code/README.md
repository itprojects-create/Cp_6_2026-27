# PneumoShift Model Source

This submission contains the essential source code for training and running the
PneumoShift chest-X-ray model.

## Files

- `train.py`: extracts frozen DenseNet-121 features and trains a logistic-regression
  classifier.
- `predict.py`: predicts pneumonia/normal or pediatric/adult for a single image.
- `requirements.txt`: Python dependencies.

## Dataset layout

Place these files/folders in the project root:

```text
Indexed Images Pneumonia 1-400/
Indexed Images Normal _400-800/
AJ_Data set Both pneumonia and normal 400 each.xlsx
outputs/splits/manifest_with_splits.csv
```

Before training, create the patient-level split using the project pipeline:

```powershell
python src/build_manifest.py
python src/split_dataset.py
```

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r submission/requirements.txt
```

## Train

Pneumonia classifier:

```powershell
python submission/train.py --task pneumonia --output submission/pneumonia_model.pkl
```

Pediatric/adult classifier:

```powershell
python submission/train.py --task pediatric_adult --output submission/pediatric_adult_model.pkl
```

The split is patient-level; the script refits only from the train and validation
rows and evaluates once on the held-out test row.

## Predict

Pneumonia:

```powershell
python submission/predict.py --image "path/to/image.jpg" --model submission/pneumonia_model.pkl
```

Pediatric/adult:

```powershell
python submission/predict.py --image "path/to/image.jpg" --model submission/pediatric_adult_model.pkl
```

## Model description

- Backbone: DenseNet-121 pretrained on ImageNet
- Feature type: 1024-dimensional frozen-backbone representation
- Classifier: standardized logistic regression
