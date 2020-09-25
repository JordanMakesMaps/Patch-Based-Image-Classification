from keras.utils.vis_utils import plot_model
from keras.layers import Flatten, Concatenate, Cropping2D
from keras.layers import Input, Dense, Activation, BatchNormalization,Dropout
from keras.layers import Conv2D, AveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
   
    
def dense_layer(input_, num_filters):
    
    x = BatchNormalization()(input_)
    x = Activation('relu')(x)
    return Conv2D(num_filters, (3, 3), padding = 'same')(x)
    
        
def dense_block(input_, num_layers, num_filters):

    for _ in range(num_layers):
        output_ = dense_layer(input_, num_filters)
        input_ = Concatenate()([output_, input_])

    return input_


def build_MDNet(num_classes, img_dim, num_pipelines, num_blocks, num_layers, num_filters, decrease_by):
    
    assert img_dim[0] == img_dim[1]
    assert num_pipelines > 1
    pipelines = []

    model_input = Input(shape = img_dim)

    for pipeline_idx in range(num_pipelines):

        new_dim = int(img_dim[0] * (pipeline_idx * decrease_by)) if pipeline_idx != 0 else img_dim[0]
        crop_by = int(new_dim/2) if pipeline_idx != 0 else 0

        x = Cropping2D(cropping = crop_by,
                        name = 'Cropping_' + str(pipeline_idx) + "_" + str(crop_by))(model_input)

        for block_idx in range(num_blocks):
            
            x = dense_block(x, num_layers, num_filters)
            
            # if not the last block, add transitional layer
            if(block_idx != num_blocks - 1):
                x = BatchNormalization()(x)
                x = Conv2D(num_filters, (1, 1), padding = 'same')(x)
                x = AveragePooling2D((2, 2), strides = (2, 2))(x)
            
        x = GlobalMaxPooling2D()(x)
        
        pipelines.append(x)
    
    x = Concatenate(axis = -1)(pipelines)
    
    x = Dropout(.5)(x)

    # Two fully connected layers
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.5)(x)
    
    x = Dense(2048)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(.5)(x)

    model_output = Dense(num_classes, activation = 'softmax')(x)

    return Model(inputs = model_input, outputs = model_output, name = "MDNet")
