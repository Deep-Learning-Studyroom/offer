import numpy as np
import pprint
def conv2x2(inputs, c_out=1, kernal_size=3, padding=0, stride=1):
    p = padding
    s = stride
    k = kernal_size
    temp = np.zeros((inputs.shape[0], inputs.shape[1], inputs.shape[2] + 2 * p,
                     inputs.shape[3] + 2 * p))
    print(temp.shape)
    temp[:, :, p:-1 * p, p:-1 * p] = inputs
    print(temp)
    h_out = (inputs.shape[2] + 2 * p - k) / s + 1
    w_out = (inputs.shape[3] + 2 * p - k) / s + 1
    print(h_out)
    print(w_out)

    # conv operation
    for i in range(0, temp.shape[2], s):
        for j in range(0, temp.shape[3], s):
            outputs[i,j] = temp


images = np.random.rand(2, 3, 7, 7)


outputs = conv2x2(images)