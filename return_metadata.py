import pandas as pd
import my
import csv
import pydicom

def load_train_excel_info():
    df = pd.read_excel('CalibrationSet_NoduleData.xlsx')
    df = df.iloc[0:10, :]
    return df

def load_test_excel_info():
    df = pd.read_excel('TestSet_NoduleData_PublicRelease_wTruth.xlsx')
    df = df.iloc[0:60,:]
    return df

def load_train_dicom_info(train_path):
        
    col = ['PatientID', 'PatientName', 'PatientAge', 'PatientSex', 'StudyTime', 'StudyDate', 'BodyPartExamined', 'ScanOptions', 'ProtocolName', 'PatientPosition', 'StudyDescription', 'ImageType']

    with open('Patient_Train_Details.csv', 'w', newline = '') as csvfile:
        fieldnames = col 
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(fieldnames)
        for i in range(10):
            ds = pydicom.dcmread(train_path[i][0])
            rows = []
            for field in fieldnames:
                x = str(ds.data_element(field))
                z = x.split(':')
                rows.append(z[1])
            writer.writerow(rows)
    df = pd.read_csv('Patient_Details.csv')
    df.columns = col = ['Scan Number', 'PatientName', 'PatientAge', 'PatientSex', 'StudyTime', 'StudyDate', 'BodyPartExamined', 'ScanOptions', 'ProtocolName', 'PatientPosition', 'StudyDescription', 'ImageType']
    for j in df.columns:
        for i in range(df.shape[0]):
            df[j][i] = df[j][i][2:-1]
    return df

def load_test_dicom_info(test_path):
        
    col = ['PatientID', 'PatientName', 'PatientAge', 'PatientSex', 'StudyTime', 'StudyDate', 'BodyPartExamined', 'ScanOptions', 'ProtocolName', 'PatientPosition', 'StudyDescription', 'ImageType']

    with open('Patient_Test_Details.csv', 'w', newline = '') as csvfile:
        fieldnames = col 
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(fieldnames)
        for i in range(60):
            ds = pydicom.dcmread(test_path[i][0])
            rows = []
            for field in fieldnames:
                x = str(ds.data_element(field))
                z = x.split(':')
                rows.append(z[1])
            writer.writerow(rows)
    df = pd.read_csv('Patient_Test_Details.csv')
    df.columns = col = ['Scan Number', 'PatientName', 'PatientAge', 'PatientSex', 'StudyTime', 'StudyDate', 'BodyPartExamined', 'ScanOptions', 'ProtocolName', 'PatientPosition', 'StudyDescription', 'ImageType']
    for j in df.columns:
        for i in range(df.shape[0]):
            df[j][i] = df[j][i][2:-1]
    return df

def return_index(typo):
    train_df  = load_train_excel_info()
    test_df  =load_test_excel_info()
    train_label = []
    test_label = []
    index_mal_train = []
    index_beg_train = []
    index_mal_test = []
    index_beg_test = []
    for i in range(10):
        if train_df['Diagnosis'][i] == 'benign':
            index_beg_train.append(i)
            train_label.append(0)
        else:
            index_mal_train.append(i)
            train_label.append(1)
    for i in range(60):
        if test_df['Final Diagnosis'][i] == 'Benign nodule':
            index_beg_test.append(i)
            test_label.append(0)
        else:
            index_mal_test.append(i)
            test_label.append(1)
    if typo == 'train':
        return train_label, index_mal_train, index_beg_train
    if typo == 'test':
        return test_label, index_mal_test, index_beg_test

def load_train_info():
    
    train_label, index_mal_train, index_beg_train = return_index('train')
    
    train_path, train_label = my.return_path('Training Set', index_mal_train, index_beg_train)
    
    df_excel = load_train_excel_info()
    df_dicom = load_train_dicom_info(train_path)
    
    df = pd.merge(df_excel, df_dicom, on = ['Scan Number'])
    return df

def load_test_info():
    
    test_label, index_mal_test, index_beg_test = return_index('test')
    path_test, label_test = my.return_path('Test Set', index_mal_test, index_beg_test)
    
    df_excel = load_test_excel_info()
    df_dicom = load_test_dicom_info(path_test)
    
    df = pd.merge(df_excel, df_dicom, on = ['Scan Number'])
    
    return df
