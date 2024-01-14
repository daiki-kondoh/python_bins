import pandas as pd
import sys
import pprint
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold


def kfold_score_RandomForestClassifier(x,y,n_split):
    cv = KFold(n_splits=n_split, random_state=1, shuffle=True)
    model = RandomForestClassifier()
    scores = cross_val_score(model, x, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores,model