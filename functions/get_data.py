import tensorflow as tf
import os
import numpy as np

def get_data(dataset, base_path):


    if not os.path.exists('%s/../../data/'%base_path):
        os.makedirs('%s/../../data/'%base_path)
    
    data = {
            'mnist': get_mnist(base_path)
        }[dataset]
    
    
    return data
    
    
def get_mnist(base_path):
    
    mnist_train, mnist_test = tf.keras.datasets.mnist.load_data(
        path='%s/../../data/mnist.npz'%base_path
    )
    return mnist_train[0][:,:,:,np.newaxis], mnist_train[1], mnist_test[0][:,:,:,np.newaxis], mnist_test[1]