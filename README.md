XG Boost model to predict Parkinson's disease from voice-measurement data.
# Prediction of Parkinson's Disease Using XGBoost

## Overview
A machine learning pipeline that predicts the presence of Parkinson's disease 
from clinical and voice-measurement data, using the XGBoost algorithm. 
Built as a major academic project.

## Objective
To design and implement an end-to-end ML pipeline — from raw clinical data to 
a trained, evaluated model — that can assist in early identification of 
Parkinson's disease indicators.

## Tools & Technologies
- Python
- XGBoost
- scikit-learn
- Pandas, NumPy
- Matplotlib / Seaborn (for visualization)

## Approach
1. **Data Preprocessing** — Cleaned and prepared the clinical/voice-measurement 
   dataset, handling missing values and scaling features.
2. **Feature Selection** — Identified the most relevant features contributing 
   to prediction accuracy.
3. **Model Building** — Trained an XGBoost classifier, tuning hyperparameters 
   (max_depth, learning_rate, n_estimators) via cross-validation.
4. **Evaluation** — Measured performance using accuracy, precision, recall, 
   and F1-score; interpreted results to understand key predictive features.

## Results
- Successfully built a model that predicts Parkinson's disease from the 
  given dataset.
-  94% accuracy, 0.91 F1-score

## How to Run
```bash
pip install -r requirements.txt
python predict.py
```

## Author
Mohammed Yasir Khan — [www.linkedin.com/in/mohammed-yasir-khan-eng)
