# from sklearn.datasets import load_iris
# iris = load_iris()
# a = iris.target[[20,40,80,130]]
# print(a)
# print(iris.target_names)
# print(iris)
# ################K-Means#################
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import k_means
# from scipy.optimize import curve_fit
#
# np.random.seed(5)
#
# centers = [[1,1], [-1,-1], [1,-1]]
# X = iris.data
# Y = iris.target
#
# estimators = {'k_means_iris_3':k_means(X,n_clusters=3),'k_means_iris_8':k_means(X,n_clusters=8),'k_means_iris_bad_init':k_means(X,n_clusters=3,n_init=1,init='random')}
#
# fignum = 1
# for name, est in estimators.items():
#
#     fig = plt.figure(fignum,figsize=(4,3))
#     plt.clf()
#     ax = Axes3D(fig, rect = [0,0,0.95,1],elev = 48, azim = 134)
#
#     print('eat=',est)
#     print('name=',name)
#     plt.cla()
#     est.curve_fit(X)
#     labels = est.labels_
#     ax.scatter(X[:,3],X[:,0],X[:,2],c = labels.astype(np.float))
#
#     ax.w_xaxis.set_ticklabels([])
#     ax.w_yaxis.set_ticklabels([])
#     ax.w_zaxis.set_ticklabels([])
#     ax.set_xlabel('Petal width')
#     ax.set_ylabel('Sepal length')
#     ax.set_zlabel('Petal length')
#     fignum = fignum + 1
#
#     # Plot the ground truth
#     fig = plt.figure(fignum, figsize=(4, 3))
#     plt.clf()
#     ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
#     plt.cla()
#
#     for name, label in [('Setosa', 0),
#                         ('Versicolour', 1),
#                         ('Virginica', 2)]:
#         ax.text3D(X[y == label, 3].mean(),
#                   X[y == label, 0].mean() + 1.5,
#                   X[y == label, 2].mean(), name,
#                   horizontalalignment='center',
#                   bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
#     # Reorder the labels to have colors matching the cluster results
#     y = np.choose(y, [1, 2, 0]).astype(np.float)
#     ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)
#
#     ax.w_xaxis.set_ticklabels([])
#     ax.w_yaxis.set_ticklabels([])
#     ax.w_zaxis.set_ticklabels([])
#     ax.set_xlabel('Petal width')
#     ax.set_ylabel('Sepal length')
#     ax.set_zlabel('Petal length')
#     plt.show()from sklearn.datasets import load_iris
# iris = load_iris()
# a = iris.target[[20,40,80,130]]
# print(a)
# print(iris.target_names)
# print(iris)
# ################K-Means#################
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.cluster import k_means
# from scipy.optimize import curve_fit
#
# np.random.seed(5)
#
# centers = [[1,1], [-1,-1], [1,-1]]
# X = iris.data
# Y = iris.target
#
# estimators = {'k_means_iris_3':k_means(X,n_clusters=3),'k_means_iris_8':k_means(X,n_clusters=8),'k_means_iris_bad_init':k_means(X,n_clusters=3,n_init=1,init='random')}
#
# fignum = 1
# for name, est in estimators.items():
#
#     fig = plt.figure(fignum,figsize=(4,3))
#     plt.clf()
#     ax = Axes3D(fig, rect = [0,0,0.95,1],elev = 48, azim = 134)
#
#     print('eat=',est)
#     print('name=',name)
#     plt.cla()
#     est.curve_fit(X)
#     labels = est.labels_
#     ax.scatter(X[:,3],X[:,0],X[:,2],c = labels.astype(np.float))
#
#     ax.w_xaxis.set_ticklabels([])
#     ax.w_yaxis.set_ticklabels([])
#     ax.w_zaxis.set_ticklabels([])
#     ax.set_xlabel('Petal width')
#     ax.set_ylabel('Sepal length')
#     ax.set_zlabel('Petal length')
#     fignum = fignum + 1
#
#     # Plot the ground truth
#     fig = plt.figure(fignum, figsize=(4, 3))
#     plt.clf()
#     ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
#     plt.cla()
#
#     for name, label in [('Setosa', 0),
#                         ('Versicolour', 1),
#                         ('Virginica', 2)]:
#         ax.text3D(X[y == label, 3].mean(),
#                   X[y == label, 0].mean() + 1.5,
#                   X[y == label, 2].mean(), name,
#                   horizontalalignment='center',
#                   bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
#     # Reorder the labels to have colors matching the cluster results
#     y = np.choose(y, [1, 2, 0]).astype(np.float)
#     ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)
#
#     ax.w_xaxis.set_ticklabels([])
#     ax.w_yaxis.set_ticklabels([])
#     ax.w_zaxis.set_ticklabels([])
#     ax.set_xlabel('Petal width')
#     ax.set_ylabel('Sepal length')
#     ax.set_zlabel('Petal length')
#     plt.show()

#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
=========================================================
K-means Clustering
=========================================================

The plots display firstly what a K-means algorithm would yield
using three clusters. It is then shown what the effect of a bad
initialization is on the classification process:
By setting n_init to only 1 (default is 10), the amount of
times that the algorithm will be run with different centroid
seeds is reduced.
The next plot displays what using eight clusters would deliver
and finally the ground truth.

"""
print(__doc__)


# Code source: Gaël Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause
#example#######################################
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
#
#
# from sklearn.cluster import KMeans
# from sklearn import datasets
#
# np.random.seed(5)
#
# centers = [[1, 1], [-1, -1], [1, -1]]
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
#
# estimators = {'k_means_iris_3': KMeans(n_clusters=3),
#               'k_means_iris_8': KMeans(n_clusters=8),
#               'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
#                                               init='random')}
#
#
# fignum = 1
# for name, est in estimators.items():
#     fig = plt.figure(fignum, figsize=(4, 3))
#     plt.clf()#clears the entire current figure
#     ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=50, azim=135)#elev存储俯仰角，azim存储x-y平面方位角
#     print (est)
#     plt.cla()# clear axis
#     est.fit(X)
#     labels = est.labels_
#
#     #ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
#
#     ax.w_xaxis.set_ticklabels([])
#     ax.w_yaxis.set_ticklabels([])
#     ax.w_zaxis.set_ticklabels([])
#     ax.set_xlabel('Petal width')
#     ax.set_ylabel('Sepal length')
#     ax.set_zlabel('Petal length')
#     fignum = fignum + 1
#
# # Plot the ground truth
# fig = plt.figure(fignum, figsize=(4, 3))
# plt.clf()
# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
#
# plt.cla()
#
# for name, label in [('Setosa', 0),('Versicolour', 1),('Virginica', 2)]:
#     ax.text3D(X[y == label, 3].mean(),
#               X[y == label, 0].mean() + 1.5,
#               X[y == label, 2].mean(), name,
#               horizontalalignment='center',
#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# # Reorder the labels to have colors matching the cluster results
# y = np.choose(y, [1, 2, 0]).astype(np.float)
# ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)
#
# ax.w_xaxis.set_ticklabels([])
# ax.w_yaxis.set_ticklabels([])
# ax.w_zaxis.set_ticklabels([])
# ax.set_xlabel('Petal width')
# ax.set_ylabel('Sepal length')
# ax.set_zlabel('Petal length')
# plt.show()
#google tech###################
# import numpy as np
# from sklearn.datasets import load_iris
# from sklearn import tree
#
# iris = load_iris()
# test_index = [0,50,100]
#
# #train data
# train_target = np.delete(iris.target, test_index)
# train_data = np.delete(iris.data, test_index,axis=0)
#
# #test data
# test_target = iris.target[test_index]
# test_data = iris.data[test_index]
#
# clf = tree.DecisionTreeClassifier()
# clf.fit(train_data,train_target)
#
# #pridict
# print(clf.predict(test_data))
# print(test_target)
# print("acc=",np.mean(clf.predict(test_data)==test_target))

# #vis 可视化
# from IPython.display import Image
# import pydotplus
# import pydot
# from IPython.display import Image
#
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = pydotplus.graph_from_dot_data(dot_data)
# graph.write_pdf("iris.pdf")
# ot_data = tree.export_graphviz(clf, out_file=None,
#                          feature_names=iris.feature_names,
#                          class_names=iris.target_names,
#                          filled=True, rounded=True,
#                          special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data)
# Image(graph.create_png())
#hist gram lecture 3
# import matplotlib.pyplot as plt
# import numpy as np
# grehounds = 500
# labs = 500
#
# grey_height = 28 + 4 * np.random.randn(grehounds)#返回500个正太分布的数构成一个数组
# lab_height = 28 + 4 * np.random.randn(labs)
#
# plt.hist([grey_height,lab_height], stacked = False, color=['r','b'])
# plt.hist([grey_height,lab_height], stacked = True, color=['r','b'])
# plt.show()
# print(grey_height)
# print(lab_height)

####################lecture4
from scipy.spatial import distance
import random

def rul(a,b):
    result = distance.euclidean(a,b)
    return  result

class LxqKNN():
    def fit(self,X_train,Y_train):
        # pass
        self.X_train = X_train
        self.Y_train = Y_train

    def predict(self,X_test):
        #pass
        prediction = []
        for row in X_test:
            label = self.closest(row)
            prediction.append(label)
        return prediction
    def closest(self,row):
        best_dist = rul(row,self.X_train[0])
        best_index = 0
        for i in range(1,len(self.X_train)):
            dist = rul(row,self.X_train[i])
            if dist < best_dist:
                dist = best_dist
                best_index = i
        return self.Y_train[best_index]

from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.cross_validation import train_test_split
import numpy as np
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=.5)#0.5~有一半用来测试一半用来训练，各75个
#tree
from sklearn import tree
my_classifier = tree.DecisionTreeClassifier()

my_classifier.fit(X_train,y_train)
result = my_classifier.predict(X_test)


acc = np.mean(result==y_test)
print('DecisionTree acc = ',acc)

# from sklearn.metrics import accuracy_score
# print(accuracy_score(result,y_test))

#KneighboursClassifier
# from sklearn.neighbors import KNeighborsClassifier
# my_classifier = KNeighborsClassifier()

##use the Kneighbours i designed:
my_classifier = LxqKNN()

my_classifier.fit(X_train,y_train)
result = my_classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print('Kneighbours acc=', accuracy_score(result,y_test))

