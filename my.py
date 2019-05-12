import numpy as np
import pandas as pd
import pydicom
import os
from tqdm import tqdm_notebook
import data_augmentation

def create_path_list(path_name, index_mal, index_beg):
    """
    index : index of last malignant
    """
    folder_path = path_name
    image_path = os.listdir(folder_path)
    
    malignant = []
    benign = []
    
    for i in index_mal:
        malignant.append(image_path[i])
    for i in index_beg:
        benign.append(image_path[i])
    
    return malignant, benign

def find_path(path_list, name, path_name):
    
    path = []
    label = []
    
    if name == 'malignant':
        l = 1
    if name == 'benign':
        l = 0
    
    for index, i in enumerate(path_list):
        path.append([])
        label.append(l)
        folderpath = path_name + '/' + str(i)
        #print(index, index, folderpath)
        for dirpath, dirname, filenames in os.walk(folderpath):
            for file in filenames:
                if '.dcm' in file.lower():
                    path[index].append(os.path.join(dirpath,file))

    return path, label

def return_path(path_name, index_mal, index_beg):
    
    malignant, benign = create_path_list(path_name, index_mal, index_beg)
    path_m, lab_m = find_path(malignant, 'malignant', path_name)
    path_b, lab_b = find_path(benign, 'benign', path_name)
    
    path = path_m + path_b
    label = lab_m + lab_b
    
    return path, label

def retrun_data(path):
    data = np.array([ [] for i in range(64*250)])
    for i in tqdm_notebook(range(len(path)), total = len(path), unit = 'i'):
        print('starting : ', i+1)
        dummy = np.array([[] for i in range(64) ])
        for j in tqdm_notebook(range(250), total = 250, unit = 'j'):
            ds = pydicom.dcmread(path[i][j])
            ds = ds.pixel_array[::8,::8]
            if j == 0:
                dummy = np.hstack((dummy, ds))
            else:
                dummy = np.vstack((dummy, ds))
        if i == 0:
            data = np.hstack((data, dummy))
        else:
            data = np.vstack((data, dummy))
        
    return data

def modified_retrun_data(path, typo):
    window = [100, 100, 100]
    gap = [45, 45, 30]
    max_window = [512, 512, 250]
    data = np.array([ [] for i in range(100*100)])
    min_z, max_z = data_augmentation.return_z(typo, 25, 100, 1)
    c_min, c_max = data_augmentation.dataAugmentation(typo, window, gap, max_window)
    for i in tqdm_notebook(range(len(path)), total = len(path), unit = 'i'):
        print('starting : ', i+1)
        a = min_z[i][0]
        b = max_z[i][0] + 1
        dummy = np.array([[] for i in range(100) ])
        for j in range(a, b):
            minx, miny, minz = c_min[i][0]
            maxx, maxy, maxz = c_max[i][0]
            #print(a, b, j)
            ds = pydicom.dcmread(path[i][j])
            ds = ds.pixel_array[minx : maxx + 1, miny : maxy + 1]
            if j == a:
                dummy = np.hstack((dummy, ds))
            else:
                dummy = np.vstack((dummy, ds))
        if i == 0:
            data = np.hstack((data, dummy))
        else:
            data = np.vstack((data, dummy))
        
    return data
