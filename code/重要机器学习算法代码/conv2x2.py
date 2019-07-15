import numpy as np
def conv2x2(input, kernal, bias=None, stride=1, padding=0):
    """
    二维卷积函数
    input:input array of shape(n, c_in, h_in, w_in)
    kernal: filters of shape(c_out, c_in, k_h, k_w)

    :return: 卷积后的图片
    """

    # 检查卷积核的c_in是否等于input的c_in
    if input.shape[1] != kernal.shape[1]:
        raise Exception("c_in of kernal is not equal to the c_in of input")

    # 检查图片尺寸
    if input.shape[2] < 2 or input.shape[3] < 2:
        raise ValueError("input value is not an array of normal picture")

    # 假设stride和padding都是默认值

    h_out = input.shape[2] - kernal.shape[2] + 1
    w_out = input.shape[3] - kernal.shape[3] + 1
    output = np.zeros((input.shape[0], kernal.shape[0], h_out, w_out))
    