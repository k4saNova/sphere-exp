import sys
sys.path.append(".")

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score

from sphexp import *

def load_data(name, scaler, n_samples=1000, fit=True):
    if name == "blobs":
        x, y = make_blobs(n_samples=1000, n_features=2,
                          cluster_std=2.0, random_state=3)
        return x, y

    elif name == "moons":
        x, y =  make_moons(1000, noise=0.3, random_state=3)
        return x, y

    else:
        raise ValueError("invalid dataset name")

    
def generate_samples(n_samples):
    """ returns samples ~ [0, 1]^2
    """
    return np.random.random((n_samples, 2))


def main():
    np.random.seed(314)
    
    fig_path = "examples/ex2/figs/"
    data_name = "moons"
    scaler = MinMaxScaler()
    x, y = load_data(data_name, scaler, fit=True)
    x = scaler.fit_transform(x)
    clf = SVC() # RandomForestClassifier() # 
    clf_name = "svc" # "rf" # 
    clf.fit(x, y)
    print(f"train score: {clf.score(x, y)}")

    # plot boundary
    plot_boundary(x, y, clf, plot_data=True,
                  show=False, save=True,
                  file_name=fig_path + f"{clf_name}-bound.png")

    N = 5000
    VERBOSE = False
    z  = generate_samples(N)
    yz = clf.predict(z)
    domain = [(0, 1), (0, 1)]
    sphexp = SphereExplainer(domain, verbose=VERBOSE)
    target_label = 0
    max_sub = 3
    max_super = 10
    sphexp.fit_subset(target_label, z, yz, max_sub)
    sphexp.fit_superset(target_label, z, yz, max_super)
    plot_circles(z, yz, sphexp.superset[target_label], clf, plot_data=False,
                 show=False, save=True, file_name=fig_path + f"{clf_name}-super.png")
    plot_circles(z, yz, sphexp.subset[target_label], clf, plot_data=False,
                 show=False, save=True, file_name=fig_path + f"{clf_name}-sub.png")

    x_sample = generate_samples(100000)
    y_sample = clf.predict(x_sample)
    hr_recall, hp_recall = sphexp.recall(x_sample, y_sample, target_label)
    hr_precision, hp_precision = sphexp.precision(x_sample, y_sample, target_label)
    hr_cov, hp_cov = sphexp.coverage(x_sample, target_label)
    print("===== High-recall =====")
    print(f"recall:    {hr_recall}")
    print(f"precision: {hr_precision}")
    print(f"coverage:  {hr_cov}")
    print("=======================")
    print()
    print("=== High-Precision ====")
    print(f"recall:    {hp_recall}")
    print(f"precision: {hp_precision}")
    print(f"coverage:  {hp_cov}")
    print("=======================")

    
if __name__ == '__main__':
    main()
