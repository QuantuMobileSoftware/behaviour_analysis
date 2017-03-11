import pandas as pd


def is_not_null(x):
    if pd.isnull(x):
        return 0
    else:
        return 1


def preprocessing(df):

    dataset = df.copy()

    # drop duplicates
    dataset = dataset.drop_duplicates()

    # features selection
    unrelevant_features = ['date', 'mezip', 'email_length', 'email_alpha_chars', 'email_numeric_chars',
                           'email_special_chars', 'age_avg', 'age_std', 'total_alive_with_name',
                           'irs_filers_count', 'agi_avg_income', 'nzip', 'nhombre', 'tid', 'contact_id', 'cupdate',
                           'especial', 'case']

    # removing unrelevant columns
    dataset.drop(unrelevant_features, inplace=True, axis=1)

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

    # processing missing numbers
    dataset[number_features] = dataset[number_features].fillna(0)

    # processing missing text data
    dataset[text_features] = dataset[text_features].fillna('miss')

    # processing data
    dataset['created_at'] = pd.to_datetime(dataset['created_at'])
    dataset['diff'] = pd.to_datetime('today') - dataset['created_at']
    dataset.drop(['created_at'], inplace=True, axis=1)
    dataset['diff'] = dataset['diff'].astype('timedelta64[D]')

    # make target label
    dataset['ccreate'] = dataset['ccreate'].apply(is_not_null)
    target_data = dataset['ccreate']
    dataset.drop(['ccreate'], inplace=True, axis=1)

    # mapping
    dataset = pd.get_dummies(dataset)

    return dataset, target_data
