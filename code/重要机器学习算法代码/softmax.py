import numpy as np


def convert_label_to_onehot(classes, labels):
    """
    classes: 类别数
    labels: array,shape=(N,)
    """
    return np.eye(classes)[labels].T


def softmax(logits):
    """logits: array, shape=(N, C)"""
    pred = np.argmax(logits, axis=1)
    return pred


if __name__ == '__main__':
    # 有四个样本，有三个类别
    logits = np.array([[0, 0.45, 0.55],
                       [0.9, 0.05, 0.05],
                       [0.4, 0.6, 0],
                       [1, 0, 0]])
    pred = softmax(logits)
    print(pred)
