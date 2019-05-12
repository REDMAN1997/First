import pandas as pd
import numpy as np
import return_metadata as rm
import my
import cv2
np.random.seed(0)

def init(col_name,typo):
    
    if typo == 'train':
        df = rm.load_train_excel_info()
        return df[col_name].as_matrix().astype(np.int32)
    if typo == 'test':
        df = rm.load_test_excel_info()
        return df[col_name].as_matrix().astype(np.int32)

def create_point(x, max_window, x_window, gap):
    a = []
    b = []
    for i in range(len(x)):
        if (max_window - x[i]) >= (x_window - gap):
            low_x = max(0, x[i] - x_window + gap)
            high_x = max(x[i] - gap, 1)
            r = np.random.randint(low = low_x, high = high_x, size = (10))
            r = list(set(r))
            min_x = r
            max_x = np.zeros(shape = len(r)) + x_window - 1
            max_x = r + max_x
            max_x = list(max_x.astype(np.int32))
            a.append(min_x)
            b.append(max_x)
        else:
            low_x = min(max_window - 1, x[i] + gap)
            high_x = min(max_window, x[i] + x_window - gap)
            r = np.random.randint(low = low_x, high = high_x, size = (10))
            r = list(set(r))
            max_x = r
            min_x = np.zeros(shape = len(r)) + x_window - 1
            min_x = r - min_x
            min_x = list(min_x.astype(np.int32))
            a.append(min_x)
            b.append(max_x)
    return a, b

def cartesian_product(m_x, m_y, m_z):
    cartesian_product_m = []
    for i in range(len(m_x)):
        cp = [[index_x, index_y, index_z] for index_x in m_x[i] for index_y in m_y[i] for index_z in m_z[i]]
        cartesian_product_m.append(cp)
    return cartesian_product_m

def augmentation(col_name, window, gap, max_window, typo):
    
    x_name, y_name, z_name = col_name
    x = init(x_name, typo)
    y = init(y_name, typo)
    z = init(z_name, typo)
    
    x_window, y_window, z_window = window
    x_gap, y_gap, z_gap = gap
    max_window_x, max_window_y, max_window_z = max_window
    min_x, max_x = create_point(x, max_window_x, x_window, x_gap)
    min_y, max_y = create_point(y, max_window_y, y_window, y_gap)
    min_z, max_z = create_point(z, max_window_z, z_window, z_gap)
    
    cartesian_product_min = cartesian_product(min_x, min_y, min_z)
    cartesian_product_max = cartesian_product(max_x, max_y, max_z)
    
    return cartesian_product_min, cartesian_product_max

def dataAugmentation(typo, window, gap, max_window):
    
    col_name = ['Nodule Center x Position', 'Nodule Center y Position*', 'Nodule Center Image']
    
    cp_min, cp_max = augmentation(col_name, window, gap, max_window, typo)
    
    return cp_min, cp_max

def return_z(typo, gap, window, num):
    
    if typo == 'train':
        df = rm.load_train_excel_info()
        train_label, index_mal_train, index_beg_train = rm.return_index('train')
        path, _ = my.return_path('/content/gdrive/My Drive/SPIE-AAPM Lung CT Challenge/Training Set', index_mal_train, index_beg_train)
    if typo == 'test':
        df = rm.load_test_excel_info()
        test_label, index_mal_test, index_beg_test = rm.return_index('test')
        path, labels_test = my.return_path('/content/gdrive/My Drive/SPIE-AAPM Lung CT Challenge/Test Set', index_mal_test, index_beg_test)
        
    a = []
    b = []

    col_name = 'Nodule Center Image'
    z = df[col_name].as_matrix().astype(np.int32)
    
    for i, p in enumerate(path):
        if (len(p) - z[i]) >= (window - gap):
            low_z = max(0, z[i] - window + gap)
            high_z = max(1, z[i] - gap)
            #print(len(p), low_z, high_z)
            r = np.random.randint(low = low_z, high = high_z, size = (num))
            r = list(set(r))
            min_z = r
            max_z = np.zeros(shape = len(r)) + window - 1
            max_z = r + max_z
            max_z = list(max_z.astype(np.int32))
            a.append(min_z)
            b.append(max_z)
        else:
            low_z = min(len(p) - 1, z[i] + gap)
            high_z = min(len(p), z[i] + window - gap)
            #print(len(p), low_z, high_z)
            r = np.random.randint(low = low_z, high = high_z, size = (num))
            r = list(set(r))
            max_z = r
            min_z = np.zeros(shape = len(r)) + window -1
            min_z = r - min_z
            min_z = list(min_z.astype(np.int32))
            a.append(min_z)
            b.append(max_z)
    return a, b

def rotate(sample, angle):
    x = sample
    for i, ds in enumerate(sample):
        rows, cols = ds.shape
        M = cv2.getRotationMatrix2D((cols//2, rows//2), angle, 1)
        dst = cv2.warpAffine(ds, M, (cols, rows))
        x[i] = dst
    return x

def flip(sample, typo):
    x = sample
    for i, ds in enumerate(sample):
        if typo == 'h':        #horitonaml flip
            ds = cv2.flip(ds, 0)
        if typo == 'v':        #vertical flip
            ds = cv2.flip(ds, 1)
        if typo == 'b':         #both flip
            ds = cv2.flip(ds, -1)
        x[i] = ds
    return x
