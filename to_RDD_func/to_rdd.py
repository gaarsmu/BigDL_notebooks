import numpy as np
from bigdl.dataset.transformer import Sample
from bigdl.util.common import *

class Error(BaseException):
    pass

init_engine()

def to_RDD(X, y, sc):
    if type(X) != np.ndarray or type(y) != np.ndarray:
        raise Error('Input arrays should be numpy ndarrays')
    if X.shape[0] != y.shape[0]:
        raise Error('Input arrays should have the same lenght')
        
    return sc.parallelize(X).zip(sc.parallelize(y)).map(
            lambda x: Sample.from_ndarray(x[0], x[1]))
