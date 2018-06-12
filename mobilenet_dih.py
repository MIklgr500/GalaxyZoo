from keras import backend as K
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import concatenate
from keras.layers import Activation
from keras.layers import Lambda
from keras.models import Model
from keras.engine.topology import get_source_inputs
from depthwise_conv2d import DepthwiseConvolution2D
from DihGroup import Dih4URandom

def get_dih_group():
    def _e(x):
        return x
    # r
    def _r90(x):
        x = K.tf.transpose(x, perm=[0,2,1,3])
        x = x[:,::-1,:]
        return x
    # r**2
    def _r180(x):
        return x[::-1,::-1,:]
    # r**3
    def _r270(x):
        x = K.tf.transpose(x, perm=[0,2,1,3])
        x = x[::-1,:,:]
        return x
    # s
    def _s(x):
        return x[::-1,:,:]
    # rs
    def _rs90(x):
        x = _s(x)
        x = K.tf.transpose(x, perm=[0,2,1,3])
        x = x[:,::-1,:]
        return x
    # r**2s
    def _rs180(x):
        x = _s(x)
        return x[::-1,::-1,:]
    # r**3s
    def _rs270(x):
        x = _s(x)
        x = K.tf.transpose(x, perm=[0,2,1,3])
        x = x[::-1,:,:]
        return x

    dih_group = {'e':_e,
                 'r90':_r90,
                 'r180':_r180,
                 'r270':_r270,
                 #'s':_s,
                 #'rs90':_rs90,
                 #'rs180':_rs180,
                 #'rs270':_rs270
                 }
    return dih_group

def Dih4(x, name):
    dih_group = get_dih_group()
    for l in dih_group:
        yield Lambda(dih_group[l], name="dih4_"+l+'_'+str(name))(x)

def MobileNetDih4(config, alpha=1.0, shape=[224,224,3]):

    img_input = Input(shape)

    x = Convolution2D(int(32 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    xd = []
    for d in Dih4(x, name='1'):
        x1 = DepthwiseConvolution2D(int(32 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(d)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Convolution2D(int(64 * alpha/4), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        xd.append(x1)
    x = concatenate(xd, axis=-1)

    xd = []
    for d in Dih4(x, name='2'):
        x1 = DepthwiseConvolution2D(int(64 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(d)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Convolution2D(int(128 * alpha/4), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        xd.append(x1)
    x = concatenate(xd, axis=-1)

    xd = []
    for d in Dih4(x, name='3'):
        x1 = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(d)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Convolution2D(int(128 * alpha/4), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        xd.append(x1)
    x = concatenate(xd, axis=-1)

    xd = []
    for d in Dih4(x, name='4'):
        x1 = DepthwiseConvolution2D(int(128 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(d)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Convolution2D(int(256 * alpha/4), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        xd.append(x1)
    x = concatenate(xd, axis=-1)

    xd = []
    for d in Dih4(x, name='5'):
        x1 = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(d)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Convolution2D(int(256 * alpha/4), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        xd.append(x1)
    x = concatenate(xd, axis=-1)

    xd = []
    for d in Dih4(x, name='6'):
        x1 = DepthwiseConvolution2D(int(256 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(d)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Convolution2D(int(512 * alpha/4), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        xd.append(x1)
    x = concatenate(xd, axis=-1)

    xd = []
    for d in Dih4(x, name='7'):
        x1 = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(d)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Convolution2D(int(512 * alpha/4), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        xd.append(x1)
    x = concatenate(xd, axis=-1)

    xd = []
    for d in Dih4(x, name='8'):
        x1 = DepthwiseConvolution2D(int(512 * alpha), (3, 3), strides=(2, 2), padding='same', use_bias=False)(d)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Convolution2D(int(1024 * alpha/4), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        xd.append(x1)
    x = concatenate(xd, axis=-1)

    for d in Dih4(x, name='9'):
        x1 = DepthwiseConvolution2D(int(1024 * alpha), (3, 3), strides=(1, 1), padding='same', use_bias=False)(d)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        x1 = Convolution2D(int(1024 * alpha/4), (1, 1), strides=(1, 1), padding='same', use_bias=False)(x1)
        x1 = BatchNormalization()(x1)
        x1 = Activation('relu')(x1)
        xd.append(x1)
    x = concatenate(xd, axis=-1)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.35)(x)

    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('tanh')(x)
    out = []
    for l in config:
        out.append(Dense(config[l], activation='linear', name=str(l))(x))

    model = Model(img_input, out, name='mobilenet_dih')
    return model
