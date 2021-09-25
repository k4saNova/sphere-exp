import sys
sys.path.append(".")
sys.path.append("examples/ex1/")
import warnings
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
from defragTrees import DefragModel

def load_data(name, n_samples=1000, fit=True):
    if name == "iris":
        iris = load_iris()
        unique, counts = np.unique(iris.target, return_counts=True)
        target_class = unique[counts.argmin()]
        target_class_name = iris.target_names[target_class] 
        print(f"{target_class_name=}")
        return iris.data, iris.target, target_class

    elif name == "wine":
        wine = load_wine()
        unique, counts = np.unique(wine.target, return_counts=True)
        target_class = unique[counts.argmin()]
        target_class_name = wine.target_names[target_class] 
        print(f"{target_class_name=}")
        return wine.data, wine.target, target_class

    elif name == "bc":
        bc = load_breast_cancer()
        unique, counts = np.unique(bc.target, return_counts=True)
        target_class = unique[counts.argmin()]
        target_class_name = bc.target_names[target_class] 
        print(f"{target_class_name=}")
        return bc.data, bc.target, target_class        
    
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
        print("target_class=spam")
        return x, y, target_class
    
    else:
        raise ValueError("specify the valid name!")


def get_clf(name):
    if name == "svc":
        return SVC()
    elif name == "mlp":
        return MLPClassifier()
    elif name == "rf":
        return RandomForestClassifier(n_estimators=100)
    else:
        raise ValueError("invalid clf name")

    
def generate_samples(n, d):
    return np.random.random((n, d))


def scores(clf, x, y, t):
    y_pred = clf.predict(x)
    hit_pred = (y_pred == t)
    hit_true = (y == t)
    return recall_score(hit_true, hit_pred), precision_score(hit_true, hit_pred), hit_pred.mean()


def true_coverage(y, t):
    return (y==t).mean()


def main():
    ## paramters
    feature_range = (0, 1)
    max_iter = 10
    n_proto = 10000
    n_samples = 50000
    dataset = "iris" # "bc" # "parkinsons" # "spam" # "iris" # "wine" #
    clf_name =  "svc" # "mlp" # "svc" # "rf"
    print(f"{max_iter=}")
    print(f"{n_proto=}")
    print(f"{n_samples=}")
    print(f"{dataset=}")
    print(f"{clf_name=}")

    
    # load a dataset
    x, y, target_class = load_data(dataset)
    dim = x.shape[1]
    domain = [feature_range for _ in range(x.shape[1])]
    recalls = {"hr": [], "hp": [], "ba": [], "dt": []}
    precisions = {"hr": [], "hp": [], "ba": [], "dt": []}
    covs = {"true": [], "hr": [], "hp": [], "ba": [], "dt": []}
        
    for i in tqdm(range(1, max_iter+1)):
        x_train, x_test, y_train, y_test =\
            train_test_split(x, y, test_size=0.2, random_state=i)
        scaler = MinMaxScaler(feature_range)
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)
        
        # classify
        clf = get_clf(clf_name)
        clf.fit(x_train, y_train)
        x_proto = generate_samples(n_proto, dim)
        y_proto = clf.predict(x_proto)
        if np.unique(y_proto).shape[0] == 1:
            print("invalid sampling")
            continue

        
        # explainer
        hs_explainer = SphereExplainer(domain)
        hs_explainer.fit(target_class, x_proto, y_proto)
        ba_tree = DecisionTreeClassifier(max_depth=4)
        ba_tree.fit(x_proto, y_proto)
        if clf_name == "rf":
            splitter = DefragModel.parseSLtrees(clf)
            de_tree = DefragModel(modeltype='classification', maxitr=100, qitr=0, tol=1e-6, restart=20, verbose=0)
            de_tree.fit(x_train, y_train, splitter, 10, fittype='FAB')
        
        # evaluation
        x_sample = generate_samples(n_samples, dim)
        y_sample = clf.predict(x_sample)
        hr_recall, hp_recall = hs_explainer.recall(x_sample, y_sample, target_class)
        hr_precision, hp_precision = hs_explainer.precision(x_sample, y_sample, target_class)
        hr_cov, hp_cov = hs_explainer.coverage(x_sample, target_class)
        ba_recall, ba_precision, ba_cov = scores(ba_tree, x_sample, y_sample, target_class)
        if clf_name == "rf":
            dt_recall, dt_precision, dt_cov = scores(de_tree, x_sample, y_sample, target_class)
            recalls["dt"].append(dt_recall)
            precisions["dt"].append(dt_precision)
            covs["dt"].append(dt_cov)
        true_cov = true_coverage(y_sample, target_class)
        
        # dump result    
        recalls["hr"].append(hr_recall)
        recalls["hp"].append(hp_recall)
        recalls["ba"].append(ba_recall)
        precisions["hr"].append(hr_precision)
        precisions["hp"].append(hp_precision)
        precisions["ba"].append(ba_precision)
        covs["hr"].append(hr_cov)
        covs["hp"].append(hp_cov)
        covs["ba"].append(ba_cov)
        covs["true"].append(true_cov)
        
    # hyouzi
    print("------")
    print("Recall")
    print(f"HR: {np.mean(recalls['hr']):.5f}")
    print(f"HP: {np.mean(recalls['hp']):.5f}")
    print(f"BA: {np.mean(recalls['ba']):.5f}")
    if clf_name == "rf":
        print(f"DT: {np.mean(recalls['dt']):.5f}")
    print("---")
    print("Precision")
    print(f"HR: {np.mean(precisions['hr']):.5f}")
    print(f"HP: {np.mean(precisions['hp']):.5f}")
    print(f"BA: {np.mean(precisions['ba']):.5f}")
    if clf_name == "rf":
        print(f"DT: {np.mean(precisions['dt']):.5f}")
    print("---")
    print("Coverage")
    print(f"True: {np.mean(covs['true']):.5f}")
    print(f"HR: {np.mean(covs['hr']):.5f}")
    print(f"HP: {np.mean(covs['hp']):.5f}")
    print(f"BA: {np.mean(covs['ba']):.5f}")
    if clf_name == "rf":
        print(f"DT: {np.mean(covs['dt']):.5f}")    
    print("------")
        
if __name__ == '__main__':
    np.random.seed(314)
    warnings.simplefilter('ignore')
    main()
