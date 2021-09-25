import sys
sys.path.append(".")

import numpy as np
from pprint import pprint
from sphexp import SphereExplainer

def test():
    x = np.random.random((10, 3))
    exp = SphereExplainer([(0,1), (0,1), (0,1)])
    res = exp.miniball(x)
    pprint(res)
    assert True is True
    
if __name__ == '__main__':
    test()
