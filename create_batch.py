import numpy as np
import my
import data_augmentation

def reduced_data_64(sample, i ,num):

    c_min, c_max = data_augmentation.dataAugmentation('train')
 
    length = len(c_min[i])
    index = np.random.randint(0, length - 1, num)
    final = np.array([[] for k in range(100 * 64)])
    
    for j in range(num):
    
        minx, miny, minz = c_min[i][index[j]]
        maxx, maxy, maxz = c_max[i][index[j]]
    
        x = sample[i, minz:maxz + 1, :,:]
        x = x.reshape(-1, 64)
        
        if j == 0:
            final = np.hstack((final, x))
        else:
            final = np.vstack((final, x))

    final = final.reshape(num, 100, 64, 64)

    return final

def create_batches(sample, labels, num_example, batch_size):

    index_i = np.random.randint(0, num_example, batch_size)
  
    label = []
  
    n = np.array([ [] for k in range(100*64)])
  
    for i, index in enumerate(index_i):
    
        x = reduced_data_64(sample, index, 1)
  
        x = x.reshape(-1, 64)
        
        label.append(labels[index])
        
        if i == 0:
            n = np.hstack((n, x))
        else:
            n = np.vstack((n, x))
    label = np.array(label)
    n = n.reshape(batch_size, 100, 64, 64)
    return index_i, label, n

def flip(typo, flag, n, sample, labels, label, i):
    
    x = data_augmentation.flip(sample[i], typo)
    x = x.reshape(-1, 100)
    label.append(labels[i])
    if flag:
        n = np.hstack((n, x))
        flag = 0
    else:
        n = np.vstack((n, x))
    return n, flag, label
    
def rotate(flag, n, sample, labels, label, i, low, high):
    
    angle = np.random.randint(low, high+1, 1)
    x = data_augmentation.rotate(sample[i], angle)
    x = x.reshape(-1, 100)
    label.append(labels[i])
    if flag:
        n = np.hstack((n, x))
        flag = 0
    else:
        n = np.vstack((n, x))
    return n, flag, label


def modified_create_batchs(sample, labels):
    
    label = []
    n = np.array([ [] for k in range(100 * 100)])
    num = sample.shape[0]
    flag = 1
    for i in range(num):
        
        n, flag, label = flip('h', flag, n, sample, labels, label, i)
        n, flag, label = flip('v', flag, n, sample, labels, label, i)
        n, flag, label = flip('b', flag, n, sample, labels, label, i)
        
        n, flag, label = rotate(flag, n, sample, labels, label, i, 15, 75)
        n, flag, label = rotate(flag, n, sample, labels, label, i, 105, 165)
        n, flag, label = rotate(flag, n, sample, labels, label, i, 195, 255)
        n, flag, label = rotate(flag, n, sample, labels, label, i, 285, 345)
        
    label = np.array(label)
    n = n.reshape(-1, 100, 100, 100)
    
    return label, n
