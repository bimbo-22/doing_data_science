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
    python venv/scripts/activate

    pip install -r requirements.txt


## File Structure
```bash
├── .gitignore   #files not to add to github
├── Dockerfile
├── README.md   # documentation file    
├── config      
    └── config.yaml
├── creditfaultprediction       # packaged project folder
    ├── __init__.py
    ├── components              # contains all required component for the project
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── data_validation.py
    │   ├── model_evaluation.py
    │   └── model_trainer.py
    ├── config
    │   ├── __init__.py
    │   └── configuration.py
    ├── constants              # contains constant variables
    │   ├── __init__.py
    │   └── training_pipeline.py    
    │   │   └── __init__.py
    ├── entity
    │   ├── __init___.py
    │   └── config_entity.py    
    ├── pipeline                # all components combined together
    │   └── __init__.py
    └── utils
    │   ├── __init__.py
    │   └── common.py           # additional functions needed throughout project
├── main.py
├── requirements.txt
├── research
    └── data_understanding.ipynb     # all research needed for the project incl. visualizations  
├── setup.py


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




