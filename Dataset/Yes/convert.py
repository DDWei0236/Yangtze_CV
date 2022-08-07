import numpy as np

file = np.load('/Users/weihaiyu/PycharmProjects/cv/Dataset/Yes/0_img.npy')
print(file)
np.savetxt('/Users/weihaiyu/PycharmProjects/cv/Dataset/0_img.txt',file)