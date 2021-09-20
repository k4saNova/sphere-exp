import sys
sys.path.append(".")

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from sphexp import SphereExplainer

def load_data(name, n_samples=1000, fit=True):
    if name == "blobs":
        x, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=2.0, random_state=3)
        return x, y

    elif name == "moons":
        x, y =  make_moons(n_samples, noise=0.3)
        return x, y

    elif name == "iris":
        iris = load_iris()
        unique, counts = np.unique(iris.target, return_counts=True)
        target_class = unique[counts.argmin()]
        # print(counts)
        # print(target_class)
        return iris.data, iris.target, target_class

    elif name == "bc":
        bc = load_breast_cancer()
        return bc.data, bc.target

    elif name == "parkinsons":
        df = pd.read_csv("data/parkinson/parkinsons.data")
        y = df["status"].values
        x = df.drop(["name", "status"], axis=1).values
        unique, counts = np.unique(y, return_counts=True)
        target_class = unique[counts.argmin()]
        return x, y, 1 # target_class

    elif name == "spam":
        data = np.loadtxt("data/spambase/spambase.data", delimiter=",")
        print(data.shape)
        x = data[:, :-1]
        y = data[:, -1].astype(int)
        unique, counts = np.unique(y, return_counts=True)
        target_class = unique[counts.argmin()]
        # print(np.unique(y,return_counts=True))
        return x, y, target_class
    
    else:
        raise ValueError("specify the valid name!")

    
def generate_samples(n, d):
    return np.random.random((n, d))
    # return scaler.transform(x)


def scores(clf, x, y, t):
    y_pred = clf.predict(x)
    hit_pred = (y_pred == t)
    hit_true = (y == t)
    return recall_score(hit_true, hit_pred), precision_score(hit_true, hit_pred), hit_pred.mean()


def main():
    ## paramters
    feature_range = (0, 1)
    max_iter = 1
    n_proto = 20000
    n_samples = 50000
    DATASETS = ["parkinsons"] # ["spam"]# ["iris", "spam"] # ["moons", "iris", "bc"]
    
    for dataset in DATASETS:
        # load a dataset
        print(f"{dataset=}")
        x, y, target_class = load_data(dataset)
        print(f"{target_class=}")
        dim = x.shape[1]
        domain = [feature_range for _ in range(x.shape[1])]
        recalls = {"hr": [], "hp": [], "ba": []}
        precisions = {"hr": [], "hp": [], "ba": []}
        covs = {"hr": [], "hp": [], "ba": []}
        
        for i in tqdm(range(1, max_iter+1)):
            # print(f"=== iteration {i} ===")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=i)
            scaler = MinMaxScaler(feature_range)
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            
            ## SVC
            # clf = SVC(kernel="poly", degree=5)
            clf = MLPClassifier()
            clf.fit(x_train, y_train)
            # make prototypes
            x_proto = generate_samples(n_proto, dim)
            y_proto = clf.predict(x_proto)
            # print(np.unique(y_proto, return_counts=True))
            # print(f"{clf.score(x_test, y_test)=:.5f}")

            # explainer
            hs_explainer = SphereExplainer(domain)
            hs_explainer.fit(target_class, x_proto, y_proto)
            ba_tree = DecisionTreeClassifier(max_depth=4)
            ba_tree.fit(x_proto, y_proto)
            
            x_sample = generate_samples(n_samples, dim)
            y_sample = clf.predict(x_sample)
            hr_recall, hp_recall = hs_explainer.recall(x_sample, y_sample, target_class)
            hr_precision, hp_precision = hs_explainer.precision(x_sample, y_sample, target_class)
            hr_cov, hp_cov = hs_explainer.coverage(x_sample, target_class)
            ba_recall, ba_precision, ba_cov = scores(ba_tree, x_sample, y_sample, target_class)
            
            recalls["hr"].append(hr_recall)
            recalls["hp"].append(hp_recall)
            recalls["ba"].append(ba_recall)
            precisions["hr"].append(hr_precision)
            precisions["hp"].append(hp_precision)
            precisions["ba"].append(ba_precision)
            covs["hr"].append(hr_cov)
            covs["hp"].append(hp_cov)
            covs["ba"].append(ba_cov)
            
        # hyouzi
        print("Recall")
        print(f"HR: {np.mean(recalls['hr']):.5f}")
        print(f"HP: {np.mean(recalls['hp']):.5f}")
        print(f"BA: {np.mean(recalls['ba']):.5f}")
        print("---")
        print("Precision")
        print(f"HR: {np.mean(precisions['hr']):.5f}")
        print(f"HP: {np.mean(precisions['hp']):.5f}")
        print(f"BA: {np.mean(precisions['ba']):.5f}")
        print("---")
        print("Coverage")
        print(f"HR: {np.mean(covs['hr']):.5f}")
        print(f"HP: {np.mean(covs['hp']):.5f}")
        print(f"BA: {np.mean(covs['ba']):.5f}")
        print("------")
        
if __name__ == '__main__':
    # load_data("parkinsons")
    # load_data("spam")
    main()
