from keras.utils.vis_utils import plot_model

from keras.layers import Flatten, Concatenate, Cropping2D
from keras.layers import Input, Dense, Activation, BatchNormalization, Dropout
from keras.layers import Conv2D, AveragePooling2D, GlobalMaxPooling2D
from keras.models import Model

import keras.backend as K
   
    
def dense_layer(input_, num_filters, dropout_rate):
    
    x = BatchNormalization()(input_)
    x = Activation('relu')(x)
    x = Conv2D(num_filters, (3, 3), padding = 'same')(x)

    if(dropout_rate):
        x = Dropout(dropout_rate)(x)
        
    return x
    
        
def dense_block(input_, num_layers, num_filters, dropout_rate):

    for _ in range(num_layers):
        output_ = dense_layer(input_, num_filters, dropout_rate)
        input_ = Concatenate(axis = -1)([output_, input_])
        
    return input_


def build_MDNet(num_classes, img_dim, num_pipelines, num_blocks, num_layers, num_filters, dropout_rate, decrease_by):
   '''
   MDNet parameters as stated in the paper:
   num_classes = 9,
   img_dim = (112, 112, 3),
   num_pipelines = 4,
   num_blocks = 4,
   num_layers = 5,
   num_filters = 8 (?),
   dropout_rate = .5,
   decrease_by = .25
   '''
   
    
    assert img_dim[0] == img_dim[1]
    assert num_pipelines > 1
    pipelines = []

    model_input = Input(shape = img_dim)

    # Number of pipelines
    for pipeline_idx in range(num_pipelines):

        new_dim = int(img_dim[0] * (pipeline_idx * decrease_by)) if pipeline_idx != 0 else img_dim[0]
        crop_by = int(new_dim/2) if pipeline_idx != 0 else 0

        x = Cropping2D(cropping = crop_by, name = 'Cropped_by_' + str(crop_by))(model_input)

        # Number of dense blocks
        for block_idx in range(num_blocks):
            
            x = dense_block(x, num_layers, num_filters, dropout_rate)
            
            # if not the last block, add transitional layer
            if(block_idx != num_blocks - 1):
                x = BatchNormalization()(x)
                z = Activation('relu')(x)
                x = Conv2D(int(K.int_shape(x)[-1] * .5), (1, 1), padding = 'same')(x)
                x = AveragePooling2D((2, 2), strides = (2, 2))(x)
            
        x = GlobalMaxPooling2D()(x)
        
        pipelines.append(x)
    
    x = Concatenate(axis = -1)(pipelines)
    
    x = Dropout(dropout_rate)(x)

    # Two fully connected layers
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    model_output = Dense(num_classes, activation = 'softmax')(x)

    return Model(inputs = model_input, outputs = model_output, name = "MDNet")
