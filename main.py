import pandas as pd
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import VotingClassifier
import preprocessing

# read dataset
ds_training = pd.read_csv('outsource_training.csv', low_memory=False)
ds_test = pd.read_csv('outsource_test1.csv', low_memory=False)

ds_training, target_training = preprocessing.preprocessing(ds_training)
ds_test, target_test = preprocessing.preprocessing(ds_test)

#
# ML algorithms
#

# dataframe for writing down results
results = pd.DataFrame()

# using tree classifier
clf_tree = tree.DecisionTreeClassifier()
try:
    clf_tree = clf_tree.fit(ds_training, target_training)
    target_predicted = clf_tree.predict(ds_test)
    print "Accuracy for tree:"
    print accuracy_score(target_test, target_predicted)
    results = results.assign(tree=target_predicted)
except Exception:
    pass

# random forest
clf_rf = RandomForestClassifier(n_estimators=10)
try:
    clf_rf = clf_rf.fit(ds_training, target_training)
    target_predicted = clf_rf.predict(ds_test)
    print "Accuracy for Random Forest:"
    print accuracy_score(target_test, target_predicted)
    results = results.assign(rf=target_predicted)
except Exception:
    pass

# gradient boosting
clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)
try:
    clf_gb = clf_gb.fit(ds_training, target_training)
    target_predicted = clf_gb.predict(ds_test)
    print "Accuracy for Gradient Boosting:"
    print accuracy_score(target_test, target_predicted)
    results = results.assign(gb=target_predicted)
except Exception:
    pass

# ensemble of three classifier

clf_tree = tree.DecisionTreeClassifier()
clf_rf = RandomForestClassifier(n_estimators=10)
clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0)

ens = VotingClassifier(estimators=[('tree', clf_tree), ('rf', clf_rf), ('gb', clf_gb)], voting='hard')
try:
    ens = ens.fit(ds_training, target_training)
    target_predicted = ens.predict(ds_test)
    print "Accuracy for ensemble:"
    print accuracy_score(target_test, target_predicted)
    results = results.assign(ensemble=target_predicted)
except Exception:
    pass

results.to_csv('results.txt', sep=',')
