import json
from sklearn.svm import SVC
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


class Model:
    def __init__(self, model=SVC(kernel="rbf", C=1, probability=True, class_weight="balanced"), threshold=0.2):
        self.model = model
        self.threshold = threshold

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save_probas(self, x):
        y_prob = self.model.predict_proba(x)[:, 1]
        x = {"predict_probas": list(y_prob), "threshold": self.threshold}
        json_object = json.dumps(x, indent=5)
        with open("probas.json", "w") as outfile:
            outfile.write(json_object)

    def score(self, x, y):
        x_prob = self.model.predict_proba(x)[:, 1]
        y_pred = x_prob > self.threshold
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        accuracy = (tn + tp) / len(y)
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = 2 * tp / (2 * tp + fp + fn)
        auc = roc_auc_score(y, self.model.predict_proba(x)[:, 1])
        return np.array([[accuracy, sensitivity, specificity, f1, auc]])
