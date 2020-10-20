import tensorflow as tf


def get_optim(optim_name, optim_config):
    optim = {
            'Adam': tf.keras.optimizers.Adam()
        }[optim_name]
    
    return optim.from_config(optim_config)