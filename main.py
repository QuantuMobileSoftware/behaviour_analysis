import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
import preprocessing

# read dataset
ds_training = pd.read_csv('outsource_training.csv', low_memory=False)
ds_test = pd.read_csv('outsource_test1.csv', low_memory=False)

#
#
#
# # drop duplicates
# ds_training = ds_training.drop_duplicates()
#
# # features selection
# unrelevant_features = ['date', 'mezip', 'email_length', 'email_alpha_chars', 'email_numeric_chars',
#                        'email_special_chars', 'age_avg', 'age_std', 'total_alive_with_name',
#                        'irs_filers_count', 'agi_avg_income', 'nzip', 'nhombre', 'tid', 'contact_id', 'cupdate', 'especial', 'case']
#
# # removing unrelevant columns
# ds_training.drop(unrelevant_features, inplace=True, axis=1)
#
# #
# # NaN values processing
# #
#
# # label relevant columns
# text_features = [
#     'sauce',
#     'us_state',
#     'us_region'
# ]
#
# number_features = [
#     'age_10pct',
#     'age_25pct',
#     'age_33pct',
#     'age_50pct',
#     'age_67pct',
#     'age_75pct',
#     'age_90pct',
#     'gender_male_prob',
#     'gender_female_prob',
#     'no_first_name_data',
#     'agi_grp1_prob',
#     'agi_grp2_prob',
#     'agi_grp3_prob',
#     'agi_grp4_prob',
#     'agi_grp5_prob',
#     'agi_grp6_prob',
#     'no_income_data',
#     'ngeo',
#     'correct_first_name'
# ]
#
# date_features = [
#     'created_at',
#     'ccreate'
# ]
#
# #
# # processing missed features
# #
#
# # processing missing numbers
# ds_training[number_features] = ds_training[number_features].fillna(0)
#
# # processing missing text data
# ds_training[text_features] = ds_training[text_features].fillna('miss')
#
# # processing data
# ds_training['created_at'] = pd.to_datetime(ds_training['created_at'])
# ds_training['diff'] = pd.to_datetime('today') - ds_training['created_at']
# ds_training.drop(['created_at'], inplace=True, axis=1)
# ds_training['diff'] = ds_training['diff'].astype('timedelta64[D]')
#
# # make target label
# ds_training['ccreate'] = ds_training['ccreate'].apply(is_not_null)
# target_data = ds_training['ccreate']
# ds_training.drop(['ccreate'], inplace=True, axis=1)
#
# # mapping
# ds_training = pd.get_dummies(ds_training)
# #
# #
# #

#
#
#
# drop duplicates
# ds_test1 = ds_test1.drop_duplicates()
#
# # features selection
# unrelevant_features = ['date', 'mezip', 'email_length', 'email_alpha_chars', 'email_numeric_chars',
#                        'email_special_chars', 'age_avg', 'age_std', 'total_alive_with_name',
#                        'irs_filers_count', 'agi_avg_income', 'nzip', 'nhombre', 'tid', 'contact_id', 'cupdate', 'especial', 'case']
#
# # removing unrelevant columns
# ds_test1.drop(unrelevant_features, inplace=True, axis=1)
#
# #
# # NaN values processing
# #
#
# # label relevant columns
# text_features = [
#     'sauce',
#     'us_state',
#     'us_region'
# ]
#
# number_features = [
#     'age_10pct',
#     'age_25pct',
#     'age_33pct',
#     'age_50pct',
#     'age_67pct',
#     'age_75pct',
#     'age_90pct',
#     'gender_male_prob',
#     'gender_female_prob',
#     'no_first_name_data',
#     'agi_grp1_prob',
#     'agi_grp2_prob',
#     'agi_grp3_prob',
#     'agi_grp4_prob',
#     'agi_grp5_prob',
#     'agi_grp6_prob',
#     'no_income_data',
#     'ngeo',
#     'correct_first_name'
# ]
#
# date_features = [
#     'created_at',
#     'ccreate'
# ]
#
# #
# # processing missed features
# #
#
# # processing missing numbers
# ds_test1[number_features] = ds_test1[number_features].fillna(0)
#
# # processing missing text data
# ds_test1[text_features] = ds_test1[text_features].fillna('miss')
#
# # processing data
# ds_test1['created_at'] = pd.to_datetime(ds_test1['created_at'])
# ds_test1['diff'] = pd.to_datetime('today') - ds_test1['created_at']
# ds_test1.drop(['created_at'], inplace=True, axis=1)
# ds_test1['diff'] = ds_test1['diff'].astype('timedelta64[D]')
#
# # make target label
# ds_test1['ccreate'] = ds_test1['ccreate'].apply(is_not_null)
# target_test_data = ds_test1['ccreate']
# ds_test1.drop(['ccreate'], inplace=True, axis=1)
#
# # mapping
# ds_test1 = pd.get_dummies(ds_test1)

#
# using ML algorithms
#

# splitting dataframe

ds_training, target_training = preprocessing.preprocessing(ds_training)
ds_test, target_test = preprocessing.preprocessing(ds_test)

#
# ML algorithms
#

# using tree classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(ds_training, target_training)

# test our model
target_predicted = clf.predict(ds_test)

print accuracy_score(target_test, target_predicted)
