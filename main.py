import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


def is_not_null(x):
    if pd.isnull(x):
        return 0
    else:
        return 1


# read dataset
ds_training = pd.read_csv('outsource_training.csv', low_memory=False)

# drop duplicates
ds_training = ds_training.drop_duplicates()

# features selection
unrelevant_features = ['date', 'mezip', 'email_length', 'email_alpha_chars', 'email_numeric_chars',
                       'email_special_chars', 'age_avg', 'age_std', 'total_alive_with_name',
                       'irs_filers_count', 'agi_avg_income', 'nzip', 'nhombre', 'tid', 'contact_id', 'cupdate', 'especial', 'case']

# removing unrelevant columns
ds_training.drop(unrelevant_features, inplace=True, axis=1)

#
# NaN values processing
#

# label relevant columns
text_features = [
    'sauce',
    'us_state',
    'us_region'
]

number_features = [
    'age_10pct',
    'age_25pct',
    'age_33pct',
    'age_50pct',
    'age_67pct',
    'age_75pct',
    'age_90pct',
    'gender_male_prob',
    'gender_female_prob',
    'no_first_name_data',
    'agi_grp1_prob',
    'agi_grp2_prob',
    'agi_grp3_prob',
    'agi_grp4_prob',
    'agi_grp5_prob',
    'agi_grp6_prob',
    'no_income_data',
    'ngeo',
    'correct_first_name'
]

date_features = [
    'created_at',
    'ccreate'
]

#
# processing missed features
#

# processing missing numbers
ds_training[number_features] = ds_training[number_features].fillna(0)

# processing missing text data
ds_training[text_features] = ds_training[text_features].fillna('miss')

# processing data
ds_training['created_at'] = pd.to_datetime(ds_training['created_at'])
ds_training['diff'] = pd.to_datetime('today') - ds_training['created_at']
ds_training.drop(['created_at'], inplace=True, axis=1)
ds_training['diff'] = ds_training['diff'].astype('timedelta64[D]')

# make target label
ds_training['ccreate'] = ds_training['ccreate'].apply(is_not_null)
target_data = ds_training['ccreate']
ds_training.drop(['ccreate'], inplace=True, axis=1)

# mapping
ds_training = pd.get_dummies(ds_training)

#
# using ML algorithms
#
# splitting dataframe
x_train, x_test, y_train, y_test = train_test_split(ds_training, target_data, test_size=0.2, random_state=42)

# using tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

# test our model
y_predicted = clf.predict(x_test)

print accuracy_score(y_test, y_predicted)
