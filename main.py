import pandas as pd
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import preprocessing

# read dataset
ds_training = pd.read_csv('outsource_training.csv', low_memory=False)
ds_test = pd.read_csv('outsource_validation.csv', low_memory=False)

ds_training, target_training = preprocessing.preprocessing(ds_training)
ds_test, target_test = preprocessing.preprocessing(ds_test)

#
# ML algorithms
#

# using tree classifier
try:
    clf_tree = tree.DecisionTreeClassifier()
    clf_tree = clf_tree.fit(ds_training, target_training)
    target_predicted = clf_tree.predict(ds_test)
    print "Accuracy for tree:"
    print len(target_predicted)
    print len(target_predicted[target_predicted==1])
    print accuracy_score(target_test, target_predicted)
except Exception:
    pass

# random forest
try:
    clf_rf = RandomForestClassifier(n_estimators=10)
    clf_rf = clf_rf.fit(ds_training, target_training)
    target_predicted = clf_rf.predict(ds_test)
    print "Accuracy for Random Forest:"
    print len(target_predicted)
    print len(target_predicted[target_predicted == 1])
    print accuracy_score(target_test, target_predicted)
except Exception:
    pass

# gradient boosting
try:
    clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=3, random_state=0).fit(ds_training, target_training)
    target_predicted = clf_gb.predict(ds_test)
    print "Accuracy for Gradient Boosting:"
    print len(target_predicted)
    print len(target_predicted[target_predicted == 1])
    print accuracy_score(target_test, target_predicted)
except Exception:
    pass
