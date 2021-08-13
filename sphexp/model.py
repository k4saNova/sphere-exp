import numpy as np
from itertools import combinations, product
from tqdm import tqdm
from sklearn.cluster import KMeans, AgglomerativeClustering
from .utils import *

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
    def __init__(self, feature_range, model=None,
                 categorical_features={},
                 n_cov=100000, random_state=314,
                 verbose=True):
        """Initialize Sphere Explainer
        it supports only numerical features.

        Args:
        feature_names (List[str]): list of feature names
        categorical_features (Dict[str, List[str]]): keys are categorical feature name,
                                                     values are the categories for the feature
        """
        self.seed = random_state
        self.num_features = len(feature_range)
        self.superset_exp = {}
        self.subset_exp = {}
        self.n_cov = n_cov
        self.n_batch = 10000
        self.domain_min = np.array([v for v, _ in feature_range])
        self.domain_max = np.array([v for _, v in feature_range])
        self.model = model
        self.verbose = verbose


    def show_message(self, msg):
        if self.verbose: print(msg)


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


    def duplicated_coverage(self, spheres):
        np.random.seed(self.seed)
        hit = 0.0
        n = 0
        while n < self.n_cov:
            xs = np.random.random((self.n_batch, self.num_features))
            for c, r in spheres.values():
                hit += (np.linalg.norm(xs - c, axis=1) < r).sum()
            n += self.n_batch
        return hit / n



    def score(self, predict_fn, target_label, n_samples=None):
        """returns recall, precision, accuracy and coverage
        score := #{z: f(x)=f(z) and z \in S} / #{z: f(x)=f(z)}
        cov   := E[f(x)=f(z)] (z sampled from the domain)
        """
        score, cov = 0., 0.
        n, m = 0, 0
        spheres = self.superset_exp[target_label]
        if n_samples is None:
            n_samples = self.n_cov

        while n < n_samples:
            xs = np.random.random((self.n_batch, self.num_features))
            pl = predict_fn(xs)
            m += xs[pl==target_label].shape[0]
            hit = np.array([False] * xs.shape[0])
            for c, r in spheres.values():
                hit = hit | (np.linalg.norm(xs - c, axis=1) < r)
            score += hit[pl==target_label].sum()
            cov += hit.sum()
            n += self.n_batch
        return score / m, cov / n



    def fit_superset(self, target_label, prototypes, labels,
                     max_sphere=10):
        """
        Args
        target_label (int, str, ..): the target label
        prototypes (2darray):
        labels (1darray): labels of prototypes that are assigned by self.predict_fn
        """

        # initialize explanation
        self.superset_exp[target_label] = None
        pos_pt = prototypes[labels==target_label]
        neg_pt = prototypes[labels!=target_label]
        cov_history, spheres_history = [], []
        coverage = self.hit_coverage


        def radius(c, xp):
            """
            c: centroid
            xp: positive prototypes in the sphere i
            """
            rn = np.linalg.norm(neg_pt-c, axis=1)
            rp_max = np.linalg.norm(xp-c, axis=1).max()
            if (rs := rn[rn >= rp_max]).shape[0] > 0:
                return rs.min()
            else:
                return rp_max


        def clusterize(x, n):
            """ return a clusterized label of x
            """
            if n == 1:
                return np.zeros(x.shape[0])
            elif n > 1:
                kmeans = KMeans(n_clusters=n).fit(x)
                return kmeans.predict(x)
            else:
                raise ValueError("set _spheres > 0")


        def update_history(spheres, cov):
            spheres_history.append(spheres)
            cov_history.append(cov)


        def main():
            for n in range(1, max_sphere):
                l_cluster = clusterize(pos_pt, n)
                unique_label = np.unique(l_cluster)

                spheres = {}
                for l in unique_label:
                    xp = pos_pt[l_cluster==l]
                    a = xp - xp.min(axis=0)
                    b = xp.max(axis=0) - xp
                    ri = np.where(a > b, a, b)
                    r_max = float("inf") # np.linalg.norm(ri, axis=1).min()
                    r_min_bound = ri.max(axis=1)

                    for i in np.argsort(r_min_bound):
                        if (r := radius(xp[i], xp)) < r_max:
                            spheres[l] = (xp[i], r)
                            r_max = r
                        else:
                            break

                cov = coverage(spheres)
                self.show_message(f"#s = {n}: cov = {cov}")

                if n > 1 and cov_history[-1] <= cov:
                    break
                update_history(spheres, cov)
            self.superset_exp[target_label] = spheres_history[-1]
            return spheres_history, cov_history

        return main()


    def fit_subset(self, target_label, prototypes, labels, max_sphere=10):
        # initialize explanation
        self.subset_exp[target_label] = None
        pos_pt = prototypes[labels==target_label]
        neg_pt = prototypes[labels!=target_label]
        cov_history, spheres_history = [], []
        coverage = self.hit_coverage

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

        spheres = {}
        hitmap = np.zeros((pos_pt.shape[0], pos_pt.shape[0]),
                          dtype=np.bool)

        for i, x in tqdm(enumerate(pos_pt)):
            hitmap[i, :] = (np.linalg.norm(pos_pt-x, axis=1) <= radius(x))
        self.show_message(f"hitmap: {hitmap.shape}")

        for n in range(max_sphere):
            idx = np.argmax(hitmap.sum(axis=1))
            c = pos_pt[idx]
            spheres[n] = (c, radius(c))
            cov = coverage(spheres)
            hitidx = (hitmap[idx] == True)
            hitmap[:, hitidx] = False
            update_history(spheres, cov)

        self.subset_exp[target_label] = spheres_history[-1]
        return spheres_history, cov_history
