import warnings
warnings.filterwarnings('ignore')

from keras.models import Model
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import AveragePooling2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, merge, Concatenate, Cropping2D 
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
import keras.backend as K

def dense_layer(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 3x3, Conv2D, optional dropout
    Args:
        ip: Input keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with batch_norm, relu and convolution2d added
    '''

    x = Activation('relu')(ip)
    x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def transition_layer(ip, nb_filter, dropout_rate=None, weight_decay=1E-4):
    ''' Apply BatchNorm, Relu 1x1, Conv2D, optional dropout and Maxpooling2D
    Args:
        ip: keras tensor
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor, after applying batch_norm, relu-conv, dropout, maxpool
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    x = Convolution2D(nb_filter, 1, 1, init="he_uniform", border_mode="same", bias=False,
                      W_regularizer=l2(weight_decay))(ip)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x


def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, weight_decay=1E-4):
    ''' Build a dense_block where the output of each dense_layer is fed to subsequent ones
    Args:
        x: keras tensor
        nb_layers: the number of layers of dense_layer to append to the model.
        nb_filter: number of filters
        growth_rate: growth rate
        dropout_rate: dropout rate
        weight_decay: weight decay factor
    Returns: keras tensor with nb_layers of dense_layer appended
    '''

    concat_axis = 1 if K.image_dim_ordering() == "th" else -1

    feature_list = [x]

    for i in range(nb_layers):
        x = dense_layer(x, growth_rate, dropout_rate, weight_decay)
        feature_list.append(x)
        x = merge(feature_list, mode='concat', concat_axis = concat_axis)
        nb_filter += growth_rate

    return x, nb_filter


def create_MDNet(nb_classes, img_dim, nb_dense_block = 3, nb_pipelines = 4, growth_rate=12, nb_filter=16, 
                 dropout_rate = .5, weight_decay=1E-4, decrease_by = .25):
    ''' Build the create_dense_net model
    Args:
        nb_classes: number of classes
        img_dim: tuple of shape (channels, rows, columns) or (rows, columns, channels)
        depth: number or layers
        nb_dense_block: number of dense blocks to add to end
        growth_rate: number of filters to add
        nb_filter: number of filters
        dropout_rate: dropout rate
        weight_decay: weight decay
        crop_by: percentage to crop by, symmetrical height and width
    Returns: keras tensor with nb_layers of dense_layer appended
    '''
    
    # House keeping
    assert img_dim[0] == img_dim[1]
    concat_axis = 1 if K.image_dim_ordering() == "th" else -1
    
    
    depth = 4
    nb_layers = int((depth - 4) / 3)

    pipelines = []

    # Model building
    model_input = Input(shape = img_dim)

    for pipeline_idx in range(nb_pipelines):

        new_dim = int(img_dim[0] * (pipeline_idx * decrease_by)) if pipeline_idx != 0 else img_dim[0]
        crop_by = int(new_dim/2) if pipeline_idx != 0 else 0

        pipeline_input = Cropping2D(cropping = crop_by)(model_input)

        # Initial convolution
        x = Convolution2D(nb_filter, 3, 3, init="he_uniform", border_mode="same", name="initial_conv2D_" + str(pipeline_idx), 
                          bias=False, W_regularizer=l2(weight_decay))(pipeline_input)

        x = BatchNormalization(mode=0, axis=concat_axis, gamma_regularizer=l2(weight_decay),
                                beta_regularizer=l2(weight_decay))(x)

        # Add dense blocks
        for block_idx in range(nb_dense_block):
            x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                       weight_decay=weight_decay)
            # add transition_layer
            x = transition_layer(x, nb_filter, dropout_rate=dropout_rate, weight_decay=weight_decay)

        # The last dense_block does not have a transition_layer
        x, nb_filter = dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=dropout_rate,
                                   weight_decay=weight_decay)
        
        x = GlobalAveragePooling2D()(x)

        pipelines.append(x)


    # Merge pipelines
    if(nb_pipelines > 1):
        x = Concatenate(axis = -1)(pipelines)
    
    x = Dropout(dropout_rate)(x)

    # Two fully connected layers
    x = Dense(2048,  W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Dense(2048,  W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)

    x = Dense(nb_classes, activation='softmax', W_regularizer=l2(weight_decay), b_regularizer=l2(weight_decay))(x)

    MDNet = Model(input = model_input, output = x, name = "MDNet")

    return MDNet
