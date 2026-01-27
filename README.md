# Doing Data Science Project
We are working with the Credit Card Fraud Prediction dataset (~555k transactions, 22+ attributes). The target variable is is_fraud (0/1), which we treat as ground truth for supervised classification. Our overall goal is to predict fraudulent transactions and understand which factors increase fraud risk.

### To install required libraries:
### if using conda:
    conda create -n creditfraud

    conda activate creditfraud

    pip install -r requirements.txt


#### if using usual way
    python -m venv venv

##### mac:
    source venv/bin/activate

 ##### windows:
    venv/scripts/activate

    pip install -r requirements.txt

### To Run the visualization 
    streamlit run app.py

### To Run pipeline:
    python main.py 

### Link to drawboard:
    https://excalidraw.com/#room=11fe17756923d6c4a728,t-7ao8OzhbM8reEgF3eC9Q


## File Structure
```bash
├── app.py
├── Artifact
│   ├── 01-06-2026-13-05-51
│   │   ├── data_ingestion
│   │   └── data_validation
│   ├── 01-07-2026-15-43-07
│   │   ├── data_ingestion
│   │   ├── data_transformation
│   │   └── data_validation
│   ├── 01-14-2026-22-21-09
│   │   └── model_trainer
│   └── 01-18-2026-15-26-35
│       ├── data_ingestion
│       ├── data_transformation
│       ├── data_validation
│       └── model_trainer
├── config
│   └── config.yaml
├── creditfraud
│   ├── __init__.py
│   ├── __pycache__
│   │   └── __init__.cpython-313.pyc
│   ├── components
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   └── model_trainer.py
│   ├── config
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── configuration.py
│   ├── constants
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── training_pipeline
│   ├── entity
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── artifact_entity.py
│   │   └── config_entity.py
│   ├── exception
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── exception.py
│   ├── logging
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── logger.py
│   ├── pipeline
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   └── training_pipeline.py
│   └── utils
│       ├── __init__.py
│       ├── __pycache__
│       ├── common.py
│       ├── main_utils
│       └── ml_utils
├── data_schema
│   └── schema.yaml
├── documentation
├── fraud_test.csv
├── logs
│   ├── 2026-01-18_15-15-12.log
│   ├── 2026-01-18_15-19-45.log
│   ├── 2026-01-18_15-26-07.log
│   └── 2026-01-18_15-26-35.log
├── main.py
├── mlflow.db
├── mlruns
│   ├── 432244404736493076
│   │   ├── 6b1da735f74d4bc7ae56fb21f340456f
│   │   ├── 821afafbcbe34c4f840de867e7875dea
│   │   ├── meta.yaml
│   │   ├── models
│   │   └── tags
│   └── models
│       └── CreditFraudModel
├── params.yaml
├── README.md
├── requirements.txt
├── research
│   ├── data_understanding.ipynb
│   ├── label_mappings.json
│   └── log_regression.ipynb
├── schema.yaml
├── setup.py
└── template.py
``` 

## Data Understanding


## At the moment
So far, we have:
 - Explored the dataset structure (no missing values, no duplicates, basic distributions of amount, categories, locations, etc.) to understand what information is available.

 - Engineered time-related features from trans_date_trans_time: hour of day (trans_time_group), month (trans_month), and weekday (trans_dayOfWeek), so we can analyse fraud patterns by time of day and day of week.
 - Converted birth dates into age and started defining age groups: 15–19, 20–29, 30–39, …, 70–79, and 80+ to compare transaction and fraud patterns across age segments.
 - Decided to keep and actively use geodata (lat/long and merch_lat/merch_long). We already visualised merchant locations on a US map and plan to analyse:
    - whether each merchant appears only once or operates multiple branches (same merchant name, different coordinates),
    - and later derive distance-based features between customer and merchant.
Next steps:
    - Finalize the age group implementation and merchant-branch analysis.
    - Perform focused EDA on fraud vs. non-fraud by age group, category, time of day, and weekday.
    -Set up the Python ML pipeline: one-hot encoding for categorical features, 80/20 train–test split (stratified by is_fraud), and first baseline models (e.g. Logistic Regression, Random Forest) with appropriate evaluation metrics for imbalanced data (precision, recall, F1, confusion matrix).




### All Model Results
all_models:
  AdaBoostClassifier:
    hyperparameters:
      algorithm: SAMME
      n_estimators: 80
    test:
      accuracy_score: 0.9507125890736342
      f1_score: 0.9713355671308879
      precision_score: 0.0628102990416811
      recall_score: 0.84472049689441
      roc_auc_score: 0.8979220529657271
    train:
      accuracy_score: 0.959544724930452
      f1_score: 0.959542433135147
      precision_score: 0.9527298613141469
      recall_score: 0.9670711376973539
      roc_auc_score: 0.959544724930452
  GradientBoostingClassifier:
    hyperparameters:
      criterion: squared_error
      loss: log_loss
      max_depth: 15
      min_samples_split: 20
      n_estimators: 200
    test:
      accuracy_score: 0.9984884474195638
      f1_score: 0.998484923897478
      precision_score: 0.8072100313479624
      recall_score: 0.7996894409937888
      roc_auc_score: 0.8994743991904732
    train:
      accuracy_score: 1.0
      f1_score: 1.0
      precision_score: 1.0
      recall_score: 1.0
      roc_auc_score: 1.0
  RandomForestClassifier:
    hyperparameters:
      max_depth: null
      max_features: 7
      min_samples_split: 20
      n_estimators: 1000
    test:
      accuracy_score: 0.9970848628805874
      f1_score: 0.996209541988139
      precision_score: 0.99375
      recall_score: 0.2468944099378882
      roc_auc_score: 0.6234441942266156
    train:
      accuracy_score: 0.9997685794659313
      f1_score: 0.9997685794535375
      precision_score: 1.0
      recall_score: 0.9995371589318626
      roc_auc_score: 0.9997685794659312
  Xgboost:
    hyperparameters:
      colsample_bytree: 0.3
      learning_rate: 0.1
      max_depth: 30
      n_estimators: 200
    test:
      accuracy_score: 0.9989023249118261
      f1_score: 0.9988252933033672
      precision_score: 0.9752577319587629
      recall_score: 0.734472049689441
      roc_auc_score: 0.8671998959367769
    train:
      accuracy_score: 1.0
      f1_score: 1.0
      precision_score: 1.0
      recall_score: 1.0
      roc_auc_score: 1.0
best_model:
  name: Xgboost
  test:
    accuracy_score: 0.9988543391156218
    f1_score: 0.9987669400503724
    precision_score: 0.9788583509513742
    recall_score: 0.718944099378882
    roc_auc_score: 0.8594419422661547
  threshold: 0.5357704162597656
  train:
    accuracy_score: 1.0
    f1_score: 1.0
    precision_score: 1.0
    recall_score: 1.0
    roc_auc_score: 1.0





