from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as nn
import tensorflow as tf
from umap.parametric_umap import ParametricUMAP, load_ParametricUMAP
from parametric_tSNE import Parametric_tSNE

def get_model(model_name, **kwargs):
    model = {
            'mnist': mnist_model,
            'cifar10': cifar10_model,
            'cifar10_vgg16': cifar10_vgg16,
            'umap': umap_model,
            'tsne': tsne_model
        }[model_name]
    
    return model(**kwargs)
    

def mnist_model(optim):
    model = Sequential()
    model.add(nn.InputLayer(input_shape=(28, 28, 1), name="in"))
    model.add(nn.Conv2D(filters=6, kernel_size=5, activation='relu', name="conv1"))
    model.add(nn.MaxPool2D(pool_size=2, name="pool1")) # 12x12x6 -> corresponds to conv1 in pytorch
    model.add(nn.Conv2D(filters=16, kernel_size=5, activation='relu', name="conv2"))
    model.add(nn.MaxPool2D(pool_size=2, name="pool2")) # 4x4x16 -> corresponds to conv2 in pytorch
    model.add(nn.Flatten(name="flat")) # 16x4x4
    model.add(nn.Dense(120, activation='relu', name="fc1"))
    model.add(nn.Dense(100, activation='relu', name="fc2"))
    model.add(nn.Dense(10, activation=None, name="classifier"))
    # model.add(nn.Softmax(name="classifier")) # is needed in keras to properly train on categorical crossentropy loss (in pytorch the crossentropy loss takes logits as inputs) -> or: can just set "from_logits=True" 
    # print(model.output_shape)
    
    model.compile(
        optimizer=optim,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model

def cifar10_model(optim):
    model = Sequential()
    model.add(nn.InputLayer(input_shape=(32, 32, 3), name="in"))
    model.add(nn.Conv2D(filters=16, kernel_size=3, activation="relu", name="conv1"))
    model.add(nn.Conv2D(filters=32, kernel_size=3, activation="relu", name="conv2"))
    model.add(nn.MaxPool2D(pool_size=2, name="pool1"))
    model.add(nn.Conv2D(filters=64, kernel_size=3, activation="relu", name="conv3"))
    model.add(nn.MaxPool2D(pool_size=2, name="pool2"))
    model.add(nn.Conv2D(filters=128, kernel_size=3, activation="relu", name="conv4"))
    model.add(nn.MaxPool2D(pool_size=2, name="pool3"))
    model.add(nn.Dropout(0.2, name="drop"))
    model.add(nn.Flatten(name="flat"))
    model.add(nn.Dense(10, activation=None, name="classifier"))

    model.compile(
        optimizer=optim,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


# adapted from: https://github.com/geifmany/cifar-vgg
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import optimizers
import numpy as np
from tensorflow.keras import regularizers

def cifar10_vgg16(optim):
    model = Sequential()
    weight_decay = 0.0005

    model.add(Conv2D(64, (3, 3), padding='same',
                     input_shape=(32, 32, 3),kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
    model.add(Activation('relu'))
    model.add(BatchNormalization())

    model.add(Dropout(0.5))
    model.add(Dense(10, name="classifier"))
#     model.add(Activation('softmax')) # we work with logits

    model.compile(
        optimizer=optim,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    
    try:
        model.load_weights("models/cifar10vgg.h5")
    except:
        print("pretrained model could not be loaded from location 'models/cifar10vgg.h5'")

    return model

    
def umap_model(optim, batch_size, epochs, verbose=False, save_path=None, config=None):
    if save_path is None:
        return ParametricUMAP(optimizer=optim,
            batch_size=batch_size,
            dims=None,
            encoder=None, # you could enter another network here
            loss_report_frequency=1,
            n_training_epochs=epochs,
            verbose=verbose)
    else:
        return load_ParametricUMAP(save_path)
    
    
def tsne_model(optim, batch_size, epochs, config, verbose=0, save_path=None):
    model = Parametric_tSNE(config['embedding_size'], 2, config['perplexities'], 
        hidden_layer_dims=config['layer_dims'], alpha=config['alpha'], optimizer=optim, 
        batch_size=batch_size, all_layers=None, do_pretrain=config['do_pretrain'], 
        beta_batch_size=config['beta_batch_size'], epochs=epochs, verbose=verbose)
    if save_path is not None:
        model.restore_model(save_path)
    return model