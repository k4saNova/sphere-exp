import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons, make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import recall_score, precision_score

from sphexp import *

def load_data(name, scaler, n_samples=1000, fit=True):
    def trans(x):
        if fit:
            return scaler.fit_transform(x)
        else:
            return scaler.transform(x)

    if name == "blobs":
        x, y = make_blobs(n_samples=1000, n_features=2,
                          cluster_std=2.0, random_state=3)
        return trans(x), y

    elif name == "moons":
        x, y =  make_moons(1000, noise=0.3)
        return trans(x), y

    else:
        return np.random.random((n_samples, 2)), None

data_name = "moons"
scaler = MinMaxScaler()
x, y = load_data(data_name, scaler, fit=True)
svc = SVC()
svc.fit(x, y)
print(f"train score: {svc.score(x, y)}")

# plot boundary
plot_boundary(x, y, svc, plot_data=True,
              show=False, save=True,
              file_name="tests/bound-t.png")

N = 100000
xp, _ = load_data("r", scaler, N)
yp = svc.predict(xp)

domain = [(0, 1), (0, 1)]
# sphere = SphereExplainer(domain, model=svc)

dt = DecisionTreeClassifier(max_depth=2)
dt.fit(xp, yp)
y_ba = dt.predict(xp)
print(dt.score(xp, yp))
print(recall_score(yp, y_ba))
print(precision_score(yp, y_ba))

# print("Superset")
# super_spheres_history, super_cov_history = sphere.fit_superset(0, xp, yp)
# for i, spheres in enumerate(super_spheres_history):
#     plot_circles(xp, yp, spheres.values(), svc, plot_data=False,
#                  show=False, save=True, file_name=f"tests/super-{i}.png")

# print("Subset")
# sub_spheres_history, sub_cov_history = sphere.fit_subset(0, xp, yp)
# for i, spheres in enumerate(sub_spheres_history):
#     plot_circles(xp, yp, spheres.values(), svc, plot_data=False,
#                  show=False, save=True, file_name=f"tests/subset-{i}.png")


    # print(f"score, cov = {sphere.score(svc.predict, 0, 100000)}")
