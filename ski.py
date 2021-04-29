import pandas as pd
from sklearn import datasets as dt
import numpy as np
from io import StringIO
iris = dt.load_iris()
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
x = pd.DataFrame(iris['data'], columns=iris['feature_names'])
y = pd.DataFrame(iris['target'], columns=['target'])
data = pd.concat([x, y], axis=1)
y1 = pd.DataFrame(pd.Series(iris['target']).map(
    dict(enumerate(iris['target_names']))), columns=['target_names'])

data = pd.concat([x, y1], axis=1)
