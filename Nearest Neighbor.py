import pickle as p
import matplotlib.pyplot as plt
import numpy as np


# NearestNeighbor class
class NearestNeighbor(object):
    def __init__(self):
        pass
    def train(self, X, y):
        """ X is N x D where each row is an example. Y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.Xtr = X
        self.ytr = y


    def predict(self, X):
        """ X is N x D where each row is an example we wish to predict label for """
        num_test = X.shape[0]#获得测试集行数即测试集图片个数
        # lets make sure that the output type matches the input type
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)#datatype 用来查看array里面的数据类型#Ypred中元素个数即为待测试的图片个数

        # loop over all test rows
        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            #distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)#self.Xtr是一个矩阵其行数为训练集图片个数，X[i, :]是第i个测试图片，两式做差得到了训练集所有图片和测试集对应位置差的绝对值，sum 的结果是各个训练集图片与测试集的对应位差的绝对值之和，最终是一个向量
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            min_index = np.argmin(distances)  # get the index with smallest distance#返回最小值对应的index 但是如果有多个最小值只返回第一个
            Ypred[i] = self.ytr[min_index]  # predict the label of the nearest example#得到的训练图片中和测试图片距离最小的图片的index，并将其存入到Ypred中
        return Ypred


def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
        datadict = p.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        Y = np.array(Y)  # 字典里载入的Y是list类型，把它变成array类型
        return X, Y


def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        label_names = p.load(f, encoding='latin1')
        names = label_names['label_names']
        return names
# load data
label_names = load_CIFAR_Labels("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/batches.meta")
imgX1, imgY1 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_1")
imgX2, imgY2 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_2")
imgX3, imgY3 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_3")
imgX4, imgY4 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_4")
imgX5, imgY5 = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/data_batch_5")
Xte_rows, Yte = load_CIFAR_batch("E:/8MachineLearningProject/HOG-SVM-classifer-master Face/cifar-10-python/cifar-10-batches-py/test_batch")

Xtr_rows = np.concatenate((imgX1, imgX2, imgX3, imgX4, imgX5))#每个图片占一列
Ytr_rows = np.concatenate((imgY1, imgY2, imgY3, imgY4, imgY5))#每个图片的label  （0-9中的一个）占一列

nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
nn.train(Xtr_rows[:1000,:], Ytr_rows[:1000])  # train the classifier on the training images and labels
Yte_predict = nn.predict(Xte_rows[:10,:])  # predict labels on the test images.Yte_predict是训练集中和测试集距离最小图片的索引
# and now print the classification accuracy, which is the average number
# of examples that are correctly predicted (i.e. label matches)
print('accuracy: %f' % (np.mean(Yte_predict == Yte[:10])))
print("Yte_predict:", Yte_predict)
print("Yte[]:",Yte[:10] )
# show a picture
image=imgX1[11,:1024].reshape(32,32)
print(image.shape)
# plt.imshow(image,cmap=plt.cm.gray)
plt.imshow(image)
plt.axis('off')    #去除图片边上的坐标轴
plt.show()
print('Done!')
print("imgY1","imgY1.shape",imgY1,imgY1.shape)#[6 9 9 ..., 1 1 5] (10000,)
print("imgX1","imgX1.shape",imgX1,imgX1.shape)
"""imgX1=[[ 59  43  50 ..., 140  84  72]
 [154 126 105 ..., 139 142 144]
 [255 253 253 ...,  83  83  84]
 ..., 
 [ 71  60  74 ...,  68  69  68]
 [250 254 211 ..., 215 255 254]
 [ 62  61  60 ..., 130 130 131]] (10000, 3072)"""
print(label_names)#['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
