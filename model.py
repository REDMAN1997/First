import keras
from keras.layers import Dropout, Dense, Conv3D, ZeroPadding3D, Add, Input, AveragePooling3D, MaxPooling3D, Activation, BatchNormalization, Flatten 
from keras.models import Model
from keras.initializers import glorot_uniform

def ConvNet(input_shape = (64, 64, 64, 1), classes = 2):
    
    X_input = Input(shape = input_shape)
    
    X = ZeroPadding3D((1, 1, 1), data_format = 'channels_last')(X_input)
    
    X = Conv3D(16, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X_input)
    X = Conv3D(16, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = MaxPooling3D((2, 2, 2), strides = 2)(X)
    
    X = Conv3D(32, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = Conv3D(32, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = MaxPooling3D((2, 2, 2), strides = 2)(X)
    
    X = Conv3D(64, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = Conv3D(64, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = MaxPooling3D((2, 2, 2), strides = 2)(X)
    
    X = Conv3D(128, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = MaxPooling3D((2, 2, 2), strides = 2)(X)
    
    X = Conv3D(256, (3, 3, 3), strides = 1, padding = 'same', data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = MaxPooling3D((2, 2, 2), strides = 2)(X)
    
    X = Conv3D(16, (3, 3, 3), strides = 1, data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X_input)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides = 2)(X)
    
    X = Conv3D(32, (3, 3, 3), strides = 1, data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides = 2)(X)
    
    X = Conv3D(64, (3, 3, 3), strides = 1, data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides = 2)(X)
    
    X = Conv3D(64, (3, 3, 3), strides = 1, data_format = 'channels_last', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides = 2)(X)
    
    X = AveragePooling3D((2, 2, 2),strides = 2,padding = 'valid')(X)
    
    X = Flatten()(X)
    
    X = Dense(32, activation = 'relu')(X)
    X = Dropout(0.2)(X)
    X = Dense(16, activation = 'relu')(X)
    X = Dropout(0.2)(X)
    X = Dense(8, activation = 'relu')(X)
    
    X = Dense(classes, activation = 'softmax')(X)
    
    model = Model(inputs = X_input, outputs = X, name = 'ResNet')
    return model
