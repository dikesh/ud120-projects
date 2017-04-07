#!/usr/bin/python


"""
    Starter code for the validation mini-project.
    The first step toward building your POI identifier!

    Start by loading/formatting the data

    After that, it's not our code anymore--it's yours!
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# first element is our labels, any added elements are predictor
# features. Keep this the same for the mini-project, but you'll
# have a different feature list when you do the final project.
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

# it's all yours from here forward!
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

y_predict = clf.predict(x_test)
print "Total People", len(y_predict)
print "Total POI", (y_predict == 1.0).sum()

# Accuracy of a Biased Identifier
y_pred = [0.] * 29
print "Score", clf.score(x_test, y_pred)

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

prcsn_score = precision_score(y_test, y_predict)
rcll_score = recall_score(y_test, y_predict)

print 'Precision Score', prcsn_score, 'Recall Score', rcll_score

predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

prcsn_score = precision_score(true_labels, predictions)
rcll_score = recall_score(true_labels, predictions)

print 'Precision Score', prcsn_score, 'Recall Score', rcll_score
