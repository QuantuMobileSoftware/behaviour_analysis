# Purchase predictor

## Description
Script for prediction purchases based on Machine Learning algorithms.

Program uses:
  - Pandas;
  - Numpy;
  - Scikit-Learn.

Program separated on two parts: preprocessing and prediction. Preprocessing includes next steps:
  1. Removing duplicates from dataset.
  2. Features selection
  3. NaN Values processing:
    a. all missing number values replaced by 0;
    b. all missing text values replaced by word ‘miss’;
  4. Processing date:
    a. Column ‘created_at’ replaced by days between today’s date and date in dataset;
    b. Column ‘ccreate’ replaced by ‘1’ or ‘0’ values (‘1’ - if cell in table not empty and ‘0’ - if cell is empty).
  5. Convert categorical variable into dummy/indicator variables.

Using ML algorithms:
  - Decision Tree;
  - Random Forest;
  - Gradient Boosting;
  - Ensemble of those classifiers.
All of them are binary classifiers, which at the end of script saving result to file ‘result.txt’.

## Installation

Installing:

```git clone https://github.com/exotikh3/behaviour_analysis.git```

Then (python 2.7 required):

```pip install -r requirements.txt```

After this, please, sure than you have necessary datasets (‘outsource_test1.csv’ and ‘outsource_training.csv’) and then run the script. Take a coffee break). Each time model rebuilds and result writes down to ‘result.txt’.

## Accuracy:

Accuracy is on level between 72% (Decision tree) up to 80% (Ensemble and Gradient Boosting).
