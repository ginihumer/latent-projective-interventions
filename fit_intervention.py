import sys
sys.path.append('Embeddings/pumap/')
sys.path.append('Embeddings/pumap/umap/')
sys.path.append('Embeddings/ptsne/')


import json
import functions
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as nn
import tensorflow.keras.backend as K
import pathlib
import os



class Latent_Interventions():
    def __init__(self, meta_data, verbose=0):
        if os.path.exists(meta_data["base_path"]):
            print("--Warning! Experiment path already exists. By calling further methods, the current files will be overwritten.--")
        else:
            os.makedirs(meta_data["base_path"])
            
        self.verbose = verbose
        self.meta_data = meta_data
            
        self.optim = functions.get_optim(meta_data["optim"], meta_data["optim_config"])
        self.X_train, self.y_train, self.X_test, self.y_test = functions.get_data(meta_data["dataset"], meta_data["base_path"])
        
        self.model_name = meta_data["model_name"]
        self.base_path = meta_data["base_path"]
        self.epochs = meta_data["epochs"]
        self.post_epochs = meta_data["post_epochs"]
        self.batch_size = meta_data["batch_size"]
        self.layer_key = meta_data["layer_key"]
        self.embedding_approach = meta_data["embedding_approach"]
        self.embedding_subset = meta_data["embedding_subset"]
        self.embedding_weight = meta_data["embedding_weight"]
        self.embedding_config = meta_data["embedding_conf"]
        self.embedding_epochs = meta_data["embedding_epochs"]
        self.embedding_batch_size = meta_data["embedding_batch_size"]
              
        
        self.contraction_factors = meta_data["contraction_factors"]
        self.shift_factors = meta_data["shift_factors"]
        
        self.classifier_model = None
        self.embedder_model = None
        self.sub_model = None
        self.logits_train = None
        self.logits_test = None
        
    def __call__(self):
        self.fit()
        
    def fit(self):
        self.save_dict("%s/meta_data.json"%self.meta_data["base_path"], self.meta_data)
        print("--fitting classifier basemodel--")
        self.fit_basemodel()
        print("--fitting embedder model--")
        self.fit_embedding()
        print("--fitting classifier with interventions--")
        self.fit_intervention()
        
    def fit_basemodel(self):
        
        self.classifier_model = functions.get_model(self.model_name, optim=self.optim)
        history = self.classifier_model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            verbose=self.verbose
        )
        
        self.classifier_model.save_weights("%s/classifiermodel-weights-before.hdf5"%self.base_path)
        self.save_dict("%s/classifiermodel-history-before.json"%self.base_path, history.history)
        
        
    def fit_embedding(self):
        self.init_logits()
        
        self.embedder_model = functions.get_model(self.embedding_approach, verbose=self.verbose, optim=self.optim, batch_size=self.embedding_batch_size, epochs=self.embedding_epochs, config=self.embedding_config)
        self.embedder_model.fit(self.logits_train[::self.embedding_subset].numpy()) # with tensorflow tensors, there is an indexing error...
        
        self.embedder_model.save("%s/embeddermodel"%self.base_path)
        self.save_dict("%s/embeddermodel-history.json"%self.base_path, self.embedder_model.encoder.history.history)
        
        
    def fit_intervention(self):
        if self.classifier_model is None:
            print("classifier model was None; load weights from ", "%s/classifiermodel-weights-before.hdf5"%self.base_path, "instead")
            self.classifier_model = functions.get_model(self.model_name, optim = self.optim)
            self.classifier_model.load_weights("%s/classifiermodel-weights-before.hdf5"%self.base_path)
        
        if self.embedder_model is None:
            print("embedder model was None; load model from ", "%s/embeddermodel"%self.base_path, "instead")
            self.embedder_model = functions.get_model(self.embedding_approach, optim=self.optim, batch_size=self.embedding_batch_size, epochs=self.embedding_epochs, save_path="%s/embeddermodel"%self.base_path)
        
        # --Train classifier in conjunction with the shifted embedding loss
        embedding_encoder = self.embedder_model.encoder
        embedding_encoder = tf.keras.Model(embedding_encoder.inputs, embedding_encoder.outputs, name="embedder") # rename the model; name is later needed
        embedding_encoder.trainable = False
        
        #flat = nn.Flatten()(self.classifier_model.get_layer(self.layer_key).output) # if conv layer is used as latent layer, we need to flatten the latent space...
        #embedder_out = embedding_encoder(flat)
        embedder_out = embedding_encoder(self.classifier_model.get_layer(self.layer_key).output)
        classifier_embedder_model = tf.keras.Model(self.classifier_model.input, [self.classifier_model.output, embedder_out])
        
        classifier_embedder_model.compile(
            optimizer=self.optim,
            loss={"classifier": tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), "embedder": tf.keras.losses.MeanSquaredError()},
            loss_weights={"classifier": (1.0 - self.embedding_weight)*2, "embedder": self.embedding_weight*2},
            metrics={"classifier":['accuracy']}
        )
        
        shifted_train = self.get_shifted_data()
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint("%s/classifiermodel-weights-after-embedding-epoch{epoch:02d}.hdf5"%self.base_path, monitor='loss', save_weights_only=True, verbose=self.verbose, save_best_only=False, save_freq="epoch")
        
        history = classifier_embedder_model.fit(
            x=self.X_train,
            y={"classifier": self.y_train, "embedder": shifted_train},
            batch_size=self.batch_size,
            epochs=self.post_epochs,
            shuffle=True,
            callbacks=[checkpoint],
            verbose=self.verbose
        )

        classifier_embedder_model.save_weights("%s/classifiermodel-weights-after-embedding.hdf5"%self.base_path)
        self.save_dict("%s/classifiermodel-history-after-embedding.json"%self.base_path, history.history)
        
        
        
        # --train classifier as usual, such that it is comparable with the combined model
        self.classifier_model = functions.get_model(self.model_name, optim = self.optim)
        self.classifier_model.load_weights("%s/classifiermodel-weights-before.hdf5"%self.base_path)
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint("%s/classifiermodel-weights-after-baseline-epoch{epoch:02d}.hdf5"%self.base_path, monitor='loss', save_weights_only=True, verbose=self.verbose, save_best_only=False, save_freq="epoch")

        history = self.classifier_model.fit(
            x=self.X_train,
            y=self.y_train,
            batch_size=self.batch_size,
            epochs=self.post_epochs,
            shuffle=True,
            callbacks=[checkpoint],
            verbose=self.verbose
        )
        self.classifier_model.save_weights("%s/classifiermodel-weights-after-baseline.hdf5"%self.base_path)
        self.save_dict("%s/classifiermodel-history-after-baseline.json"%self.base_path, history.history)


    def init_logits(self, refresh=False): # calculate logits from a certain layer in the classifier model
        if self.logits_train is None or refresh:
            if self.classifier_model is None:
                print("classifier model was None; loaded weights from ", "%s/model-weights-before.hdf5"%self.base_path, "instead")
                self.classifier_model = functions.get_model(self.model_name, optim = self.optim)
                self.classifier_model.load_weights("%s/classifiermodel-weights-before.hdf5"%self.base_path)
                
            # get outputs from a certain layer
            self.sub_model = tf.keras.Model(inputs=self.classifier_model.inputs, outputs=self.classifier_model.get_layer(self.layer_key).output)
            
            logits_train = self.sub_model(self.X_train)
            logits_test = self.sub_model(self.X_test)
            
            self.logits_train = tf.reshape(logits_train, [len(logits_train), -1]) # need to reshape to have only 2 dimensions
            self.logits_test = tf.reshape(logits_test, [len(logits_test), -1])
            
            self.embedding_config["embedding_size"] = self.logits_train.shape[1]
        
        
        
    def get_shifted_data(self):
        self.init_logits()
            
        projected_train = self.embedder_model.encoder(self.logits_train)
#         projected_test = self.embedder_model.encoder(self.logits_test)
        
        labels_train = self.y_train
        shifted_train = projected_train.numpy()
        distinct_labels = list(set(labels_train))

        for k in self.contraction_factors:
            if k == "all":
                contraction_factor = self.contraction_factors[k]
                for i in range(len(distinct_labels)):
                    l = distinct_labels[i]
                    shifted_train[labels_train == l] = (contraction_factor[0] * shifted_train[(labels_train == l)]) + contraction_factor[1] * shifted_train[(labels_train == l)].mean(axis=0)
            else:
                shifted_train[labels_train == k] = (contraction_factor[0] * shifted_train[(labels_train == k)]) + contraction_factor[1] * shifted_train[(labels_train == k)].mean(axis=0)
            
        return shifted_train
        
    def save_dict(self, file_path, data):
        file = open(file_path, "w")
        json.dump(data, file)
        file.close()
        print("saved file to", file_path)
        
    def load_models(self):
        self.classifier_model = functions.get_model(self.model_name, optim = self.optim)
        self.classifier_model.load_weights("%s/classifiermodel-weights-before.hdf5"%self.base_path)
        
        self.embedder_model = functions.get_model(self.embedding_approach, optim=self.optim, batch_size=self.embedding_batch_size, epochs=self.embedding_epochs, save_path="%s/embeddermodel"%self.base_path)
        
        