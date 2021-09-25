import numpy as np
from itertools import combinations, product
from sklearn.cluster import KMeans
from sklearn.metrics import recall_score, precision_score
from .utils import *
from .miniball import miniball


# util function
def normalize(v):
    if len(v.shape) == 1:
        # vector
        nv = np.linalg.norm(v)
    elif len(v.shape) == 2:
        # vectors
        nv = np.linalg.norm(v, axis=1)
        nv[nv==0] = 1
        nv = nv.reshape(-1, 1)
    return v / nv


class SphereExplainer(object):
    def __init__(self, feature_range,
                 categorical_features={},
                 n_cov=100000, random_state=314,
                 verbose=False):
        """Initialize Sphere Explainer
        it supports only numerical features.

        Args:
        feature_names (List[str]): list of feature names
        categorical_features (Dict[str, List[str]]): keys are categorical feature name,
                                                     values are the categories for the feature
        """
        self.seed = random_state
        self.num_features = len(feature_range)
        self.superset = {}
        self.subset = {}
        self.n_cov = n_cov
        self.n_batch = 10000
        self.domain_min = np.array([v for v, _ in feature_range])
        self.domain_max = np.array([v for _, v in feature_range])
        self.verbose = verbose


    def show_message(self, msg):
        if self.verbose: print(msg)


    def miniball(self, x):
        vector = x.astype(np.double)
        return miniball(vector)


    def hit_coverage(self, spheres):
        np.random.seed(self.seed)
        hit = 0.0
        n = 0
        while n < self.n_cov:
            xs = np.random.random((self.n_batch, self.num_features))
            h = np.zeros(xs.shape[0], dtype=np.bool)
            for c, r in spheres.values():
                h = h | (np.linalg.norm(xs - c, axis=1) <= r)
            hit += h.sum()
            n += self.n_batch
        return hit / n


    def check_hit(self, x, target_label):
        hit_superset = np.zeros(x.shape[0], dtype=bool)
        hit_subset = np.zeros(x.shape[0], dtype=bool)

        for c, r in self.superset[target_label].values():
            hit_superset |= (np.linalg.norm(x-c, axis=1) < r)
        for c, r in self.subset[target_label].values():
            hit_subset |= (np.linalg.norm(x-c, axis=1) < r)
        return hit_superset, hit_subset


    def recall(self, x, y_true, target_label):
        hit_superset, hit_subset = self.check_hit(x, target_label)
        hit_true = (y_true == target_label)
        return recall_score(hit_true, hit_superset), recall_score(hit_true, hit_subset)


    def precision(self, x, y_true, target_label):
        hit_superset, hit_subset = self.check_hit(x, target_label)
        hit_true = (y_true == target_label)
        return precision_score(hit_true, hit_superset), precision_score(hit_true, hit_subset)


    def coverage(self, x, target_label):
        hit_superset, hit_subset = self.check_hit(x, target_label)
        return hit_superset.mean(), hit_subset.mean()


    def fit_superset(self, target_label, prototypes, labels, max_sphere=10):
        """
        Args
        target_label (int, str, ..): the target label
        prototypes (2darray):
        labels (1darray): labels of prototypes that are assigned by self.predict_fn
        """


        def clusterize(x, n):
            """ return a clusterized label of x
            """
            if n == 1:
                return np.zeros(x.shape[0])
            elif n > 1:
                kmeans = KMeans(n_clusters=n).fit(x)
                return kmeans.predict(x)
            else:
                raise ValueError("set n_spheres > 0")


        def update_history(spheres, cov):
            spheres_history.append(spheres)
            cov_history.append(cov)


        ###### main ######
        # initialize explanation
        self.superset[target_label] = None
        pos_pt = prototypes[labels==target_label]
        neg_pt = prototypes[labels!=target_label]
        cov_history, spheres_history = [], []

        for n in range(1, max_sphere):
            if pos_pt.shape[0] <= n:
                self.show_message("invalid proto")
                break
            l_cluster = clusterize(pos_pt, n)
            unique_label = np.unique(l_cluster)

            spheres = {}
            for l in unique_label:
                xp = pos_pt[l_cluster==l]
                if xp.shape[0] < 3:
                    continue
                miniball = self.miniball(xp)
                spheres[l] = (miniball["center"], miniball["radius"])

            cov = self.hit_coverage(spheres)
            self.show_message(f"#s = {n}: cov = {cov}")

            if n > 1 and cov_history[-1] <= cov:
                self.show_message("found minimal coverage")
                break
            update_history(spheres, cov)
        self.superset[target_label] = spheres_history[-1]
        return spheres_history, cov_history



    def fit_subset(self, target_label, prototypes, labels, max_sphere=10):
        ##### functions
        def radius(c):
            """
            c: centroid
            xp: positive prototypes in the sphere i
            """
            rn_min = np.linalg.norm(neg_pt-c, axis=1).min()
            rp = np.linalg.norm(pos_pt-c, axis=1)
            if (rs := rp[rp < rn_min]).shape[0] > 0:
                return rs.max()
            else:
                return 0

        def update_history(spheres, cov):
            spheres_history.append(spheres)
            cov_history.append(cov)


        ##### main #####
        # initialize explanation
        self.subset[target_label] = None
        pos_pt = prototypes[labels==target_label]
        neg_pt = prototypes[labels!=target_label]
        cov_history, spheres_history = [], []

        spheres = {}
        hitmap = np.zeros((pos_pt.shape[0], pos_pt.shape[0]),
                          dtype=np.bool)

        for i, x in enumerate(pos_pt):
            hitmap[i, :] = (np.linalg.norm(pos_pt-x, axis=1) <= radius(x))
        self.show_message(f"hitmap: {hitmap.shape}")

        for n in range(max_sphere):
            idx = np.argmax(hitmap.sum(axis=1))
            c = pos_pt[idx]
            spheres[n] = (c, radius(c))
            cov = self.hit_coverage(spheres)
            hitidx = (hitmap[idx] == True)
            hitmap[:, hitidx] = False
            update_history(spheres, cov)
            if not hitmap.sum():
                self.show_message(f"all samples are covered")
                break

        self.subset[target_label] = spheres_history[-1]
        return spheres_history, cov_history


    def fit(self, target_label, prototypes, labels,
            max_sphere_super=10, max_sphere_sub=10):
        self.fit_subset(target_label, prototypes, labels, max_sphere_sub)
        self.fit_superset(target_label, prototypes, labels, max_sphere_super)
