import numpy as np
def mean_filter(image, kernel_size):
    """
    :param image: 图像数据 numpy array
    :kernel_size: 滤波器大小，奇数
    :return: 滤波后的图像
    """

    k = (kernel_size - 1) / 2
    height, width = image.shape[0], image.shape[1]
    for row in range(1, height - 1):
        for col in range (1, width - 1):
            # get the area to be filtered
            filtered_area = image[row - k: row + k + 1, col - k: col + k + 1]
            filtered_area = filtered_area.flatten()
            filtered_area = np.sort(filtered_area)
            res = filtered_area[(kernel_size ** 2 - 1) // 2]
            image[row, col] = res

    return image
