import numpy as np
import operator


class NearestNeibor(object):
    def __init__(self):
        pass

    def train(self, x, y):
        self.Xtr = x
        self.Ytr = y

    def predict(self, x, k):
        num_test = x.shape[0]
        Ypred = np.zeros(num_test, dtype=self.Ytr.dtype)

        for i in range(num_test):
            distances = np.sqrt(np.sum(np.square(self.Xtr - x[i, :]), axis=1))
            sorteddistances = np.argsort(distances)
            classCount = {}
            for j in range(k):
                currentLable = self.Ytr[sorteddistances[j]]
                classCount[currentLable] = classCount.get(currentLable, 0) + 1
            sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
            Ypred[i] = sortedClassCount[0][0]

        return Ypred


if __name__ == '__main__':
    Xtr = np.zeros((50000, 3072))  # 假设是训练数据Cifar-10，有50000份，每个数据有28*28*3
    Ytr = np.zeros((50000,))  # label

    #  测试数据
    Xte = np.zeros((100, 3072))
    Yte = np.zeros((100,))

    #建立模型
    model = NearestNeibor()
    model.train(Xtr, Ytr)
    Ypred = model.predict(Xte, 5)
    print(Ypred)
    print(Yte)
