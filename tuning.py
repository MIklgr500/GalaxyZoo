import os
import pandas as pd
import numpy as np
from mobilenet import MobileNet
from mobilenet_dih import MobileNetDih4
from mobilenet_dih_r import MobileNetDR
from keras.optimizers import SGD, Adam, Adadelta
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,CSVLogger
from keras import backend as K
from keras import metrics
from keras import losses
from keras.utils import to_categorical
import imageio
from skimage.transform import resize as imgresize

IMG_DIR = 'input/images_training_rev1'
CSV_PATH = 'input/training_solutions_rev1.csv'
BATCH_SIZE = 8

def train_valid_split(df:pd.DataFrame, train_size=0.7):
    n_data = len(df)
    train_length, valid_length = round(n_data*train_size), (n_data-round(n_data*train_size))//2
    print('N train samples: {0} \n N valid samples: {1}'.format(train_length, valid_length))
    train_df, valid_df = df.loc[:train_length, :], df.loc[train_length:train_length+valid_length, :]
    return train_df, valid_df

def csv_spliter(df:pd.DataFrame):
    # add 1 more subclass for all class
    class_dict ={
        'class1':['Class1.1', 'Class1.2', 'Class1.3'],
        'class2':['Class2.1', 'Class2.2'],
        'class4':['Class4.1', 'Class4.2'],
        'class5':['Class5.1', 'Class5.2', 'Class5.3', 'Class5.4'],
        'class6':['Class6.1', 'Class6.2'],
        'class7':['Class7.1', 'Class7.2', 'Class7.3'],
        'class8':['Class8.1', 'Class8.2', 'Class8.3', 'Class8.4', 'Class8.5', 'Class8.6', 'Class8.7'],
        'class9':['Class1.1', 'Class9.2', 'Class9.3'],
        'class10':['Class10.1', 'Class10.2', 'Class10.3'],
        'class11':['Class11.1', 'Class11.2', 'Class11.3', 'Class11.4', 'Class11.5', 'Class11.6']
        }

    # extracting file names
    fname = df['GalaxyID'].values
    # prob --> prediction
    df = df.iloc[:, 1:]

    # forming target array and config dict
    target = []
    tconfig = {}
    print('Number subclass in class')
    for l in class_dict:
        target.append(df[class_dict[l]].values)
        tconfig[l] = len(class_dict[l])
        print('####################################')
        print('{0}: {1}'.format(l, tconfig[l]))
    return fname, target, tconfig

def load_image(filename, cshape=[324, 324, 3], rshape=[224, 224, 3]):
    path = os.path.join(IMG_DIR, str(filename)+'.jpg')
    img = imageio.imread(path)
    rx, ry = np.random.randint(-2,high=2, size=2).tolist()
    init_shape = img.shape
    zx, zy = init_shape[0]//2, init_shape[1]//2
    dx, dy = cshape[0]//2, cshape[1]//2
    img = img[zx-dx+rx:zx+dx+rx, zy-dy+ry:zy+dy+ry, :]
    img= imgresize(img, output_shape=rshape)
    return img/255

def generator(filenames, target, batch_size):
    while True:
        img_batch = []
        target_batch_dict = {i:[] for i in range(len(target))}
        i = np.random.choice(len(filenames)-1, batch_size)
        for j in i:
            fn = filenames[j]
            for k in range(len(target)):
                target_batch_dict[k].append(target[k][j])
            img_batch.append(load_image(filenames[j]))

        img_batch = np.array(img_batch)
        target_batch = []
        for k in range(len(target)):
            target_batch.append(np.array(target_batch_dict[k]))
        yield [img_batch, target_batch]

def  get_gen_tconfig(csv_path=CSV_PATH, batch_size=BATCH_SIZE):
    df = pd.read_csv(csv_path)
    train_df, valid_df = train_valid_split(df)
    train_fname, train_target, tconfig = csv_spliter(train_df)
    valid_fname, valid_target, _ = csv_spliter(valid_df)
    train_gen = generator(train_fname, train_target, batch_size=batch_size)
    valid_gen = generator(valid_fname, valid_target, batch_size=batch_size)
    return train_gen, valid_gen, tconfig

def get_callbacks(filepath, patience=1):
    mcp = ModelCheckpoint(filepath+'.h5',
                          monitor='val_loss',
                          verbose=1,
                          save_best_only=True,
                          save_weights_only=False,
                          mode='min',
                          period=1)
    rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=patience, min_lr=1e-8)
    csv_log = CSVLogger(filename=filepath+'.csv')
    return [mcp, rlr, csv_log]

def training_model(model_name='mobilenet'):
    train_gen, valid_gen, tconfig = get_gen_tconfig()
    callbacks = get_callbacks('mobilenet05_short_adam03_dr35_v3', patience=4)
    if model_name == 'mobilenet':
        print('MobileNet')
        model = MobileNet(config=tconfig, alpha=1.0)
        model.summary()
    elif model_name =='mobilenet_dih':
        print('MobileNetDih')
        model = MobileNetDih4(config=tconfig, alpha=1)
        model.summary()
    elif model_name =='mobilenet_dih_r':
        print('MobileNetDihR')
        model = MobileNetDR(config=tconfig, alpha=0.5)
        model.summary()
    opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    #opt = Adadelta(lr=1e-1, rho=0.95, decay=0.1)
    #opt = SGD(lr=1e-7, momentum=0.9, decay=0., nesterov=True)

    model.compile(optimizer=opt,
                  loss='mse',
                  metrics=['mae', 'mse'])
    #model.load_weights('mobilenet_05shortd01_catcros_resize_b16.hdf5')
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=1000,
                        epochs=40,
                        validation_data=valid_gen,
                        verbose=2,
                        validation_steps=500,
                        callbacks=callbacks)
    #opt = Adam(lr=1e-3, beta_1=0.9, beta_2=0.999)
    #opt = Adadelta(lr=1e-1, rho=0.95, decay=0.1)
    """
    opt = SGD(lr=0.05, momentum=0.9, nesterov=True)
    callbacks = get_callbacks('mobilenet05_halfdih4_adam03_dr35_v3_sgd01', patience=4)
    model.compile(optimizer=opt,
                  loss='mse',
                  metrics=['mae', 'mse'])

    model.load_weights('mobilenet05_halfdih4_adam03_dr35_v3.h5')
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=1000,
                        epochs=10,
                        validation_data=valid_gen,
                        verbose=2,
                        validation_steps=500,
                        callbacks=callbacks)
    """
