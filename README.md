import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("framingham.csv")

data = data.dropna()


X = data.drop("TenYearCHD", axis=1)
y = data["TenYearCHD"]

model = LogisticRegression(max_iter=1000)


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []

for train_index, test_index in skf.split(X, y):
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

print("Fold Accuracies:", accuracies)
print("Mean Accuracy:", np.mean(accuracies))
