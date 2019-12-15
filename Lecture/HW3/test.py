# importing libraries
import numpy as np
from PIL import Image
import os
from numpy import linalg as LA
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import pandas as pd
from PIL import Image
import glob
from sklearn.decomposition import PCA

scalar = preprocessing.StandardScaler()  # Standardize features by removing the mean and scaling to unit variance


print(1)

# --> import
all_data = np.arange(77760)

for filename in glob.glob('yalefaces/*.*'):  # assuming gif

    img = Image.open(filename)  # 打开图像
    im = np.asarray(img, dtype='float64')
    im = np.ndarray.flatten(im)
    all_data = np.vstack([all_data, im])

DF = pd.DataFrame(data=all_data[0:, 0:])  # 1st row as the column names
DF = DF.iloc[1:]

print(2)
#############################################   Normalization
import sys
import numpy as np

pd.set_option('display.max_colwidth', -1)

np.set_printoptions(threshold=sys.maxsize)

test = np.arange(77760)

for index, X in DF.iterrows():
    X = (X - X.min()) / (X.max() - X.min())
    X = X - X.mean()
    X = X.fillna(X.mean())

    test = np.vstack([test, X])

test = pd.DataFrame(data=test[1:, 0:])


print(test.shape)




print("SVD finished")







'''
print("CA  30")
############################################   PCA  30
from sklearn.decomposition import PCA

pca = PCA(n_components=30)
pca.fit(test)

X_trans = pca.transform(test)

eigenvalues = pca.explained_variance_

index = np.arange(0, 30, 1)
index.size

import matplotlib.pyplot as plt

plt.scatter(index,eigenvalues)
plt.ylabel('')
# plt.show()
'''

print("CA  165")
############################################   PCA  165
pca_all = PCA(n_components=165)
pca_all.fit(test)

X_trans_all = pca_all.transform(test)

eigenvalues_all = pca_all.explained_variance_

index = np.arange(0, 165, 1)

import matplotlib.pyplot as plt

plt.scatter(index,eigenvalues_all)

plt.ylabel('some numbers')

plt.show()

print("50% Energy")
############################################  50% Energy
sum = 0

for i in eigenvalues_all:
    sum = sum + i

print("sum is " + str(sum))

count = 0

half_energy = 0
for i in eigenvalues_all:

    half_energy = half_energy + i

    print(half_energy)

    if (half_energy > sum / 2):
        count = count + 1
        print("50% energy from capturing : " + str(count) + " component")
        break

    count = count + 1

############################################  10 Eigenfaces





