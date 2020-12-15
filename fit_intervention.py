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


def normalize(x, mean=0, std=1):
    return (x-mean)/(std+1e-7)

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
        
        if 'normalize' in meta_data.keys() and meta_data["normalize"] is not None:
            self.X_train_default = self.X_train
            self.X_test_default = self.X_test
            self.X_train = normalize(self.X_train, mean=meta_data["normalize"]["mean"], std=meta_data["normalize"]["std"])
            self.X_test = normalize(self.X_test, mean=meta_data["normalize"]["mean"], std=meta_data["normalize"]["std"])
        
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
        self.embedding_optim = functions.get_optim(meta_data["embedding_optim"], meta_data["embedding_optim_config"])
        
              
        
        self.contraction_factors = meta_data["contraction_factors"]
        self.shift_factors = meta_data["shift_factors"]
        
        tf.random.set_seed(meta_data["experiment_number"])
        
        self.classifier_model = None
        self.embedder_model = None
        self.sub_model = None
        self.logits_train = None
        self.logits_test = None
        
    def __call__(self):
        self.fit()
        
    def save_meta_data(self):
        self.save_dict("%s/meta_data.json"%self.meta_data["base_path"], self.meta_data)
        
    def fit(self):
        self.save_meta_data()
        print("--fitting classifier basemodel--")
        self.fit_basemodel()
        print("--fitting embedder model--")
        self.fit_embedding()
        print("--fitting classifier with interventions--")
        self.fit_intervention()
        
    def fit_basemodel(self):
        
        self.classifier_model = functions.get_model(self.model_name, optim=functions.get_optim(self.meta_data["optim"], self.meta_data["optim_config"]))
        history = self.classifier_model.fit(
            self.X_train,
            self.y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=True,
            verbose=self.verbose
        )
        
        self.classifier_model.save("%s/classifiermodel-weights-before.h5"%self.base_path)
#         self.classifier_model.save_weights("%s/classifiermodel-weights-before.hdf5"%self.base_path)
        self.save_dict("%s/classifiermodel-history-before.json"%self.base_path, history.history, to_float=True)
        
        
    def fit_embedding(self):
        self.init_logits()
        
        self.embedder_model = functions.get_model(self.embedding_approach, verbose=self.verbose, optim=self.embedding_optim, batch_size=self.embedding_batch_size, epochs=self.embedding_epochs, config=self.embedding_config)
        self.embedder_model.fit(self.logits_train[::self.embedding_subset]) # with tensorflow tensors, there is an indexing error...
        
        self.embedder_model.save("%s/embeddermodel"%self.base_path)
        #print(self.embedder_model.encoder.history)
        #self.save_dict("%s/embeddermodel-history.json"%self.base_path, self.embedder_model.encoder.history.history, to_float=True)
        
        
    def fit_intervention(self):
        if self.classifier_model is None:
            print("classifier model was None; load model from ", "%s/classifiermodel-weights-before.h5"%self.base_path, "instead")
            self.classifier_model = functions.get_model(self.model_name, optim = functions.get_optim(self.meta_data["optim"], self.meta_data["optim_config"]))
            self.classifier_model.load_weights("%s/classifiermodel-weights-before.h5"%self.base_path)
#             self.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-before.h5"%self.base_path)

        
        if self.embedder_model is None:
            print("embedder model was None; load model from ", "%s/embeddermodel"%self.base_path, "instead")
            self.embedder_model = functions.get_model(self.embedding_approach, optim = functions.get_optim(self.meta_data["optim"], self.meta_data["optim_config"]), batch_size=self.embedding_batch_size, epochs=self.embedding_epochs, save_path="%s/embeddermodel"%self.base_path, config=self.embedding_config)
        
        # --Train classifier in conjunction with the shifted embedding loss
        embedding_encoder = self.embedder_model.encoder
        embedding_encoder = tf.keras.Model(embedding_encoder.inputs, embedding_encoder.outputs, name="embedder") # rename the model; name is later needed
        embedding_encoder.trainable = False
        
        flat = nn.Flatten()(self.classifier_model.get_layer(self.layer_key).output) # if conv layer is used as latent layer, we need to flatten the latent space...
        embedder_out = embedding_encoder(flat)
        #embedder_out = embedding_encoder(self.classifier_model.get_layer(self.layer_key).output)
        classifier_embedder_model = tf.keras.Model(self.classifier_model.input, [self.classifier_model.output, embedder_out])
        
        classifier_embedder_model.compile(
            optimizer= functions.get_optim(self.meta_data["optim"], self.meta_data["optim_config"]),
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

        classifier_embedder_model.save("%s/classifiermodel-weights-after-embedding.h5"%self.base_path)
#         classifier_embedder_model.save_weights("%s/classifiermodel-weights-after-embedding.hdf5"%self.base_path)
        self.save_dict("%s/classifiermodel-history-after-embedding.json"%self.base_path, history.history, to_float=True)
        
        
        
        # --train classifier as usual, such that it is comparable with the combined model
        self.classifier_model = functions.get_model(self.model_name, optim =  functions.get_optim(self.meta_data["optim"], self.meta_data["optim_config"])) # need to reinitialize the compiler, since we also do this with the "classifier_embedder_model -> comparability
        self.classifier_model.load_weights("%s/classifiermodel-weights-before.h5"%self.base_path)
#         self.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-before.h5"%self.base_path)

        
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
        
        self.classifier_model.save("%s/classifiermodel-weights-after-baseline.h5"%self.base_path)
#         self.classifier_model.save_weights("%s/classifiermodel-weights-after-baseline.hdf5"%self.base_path)
        self.save_dict("%s/classifiermodel-history-after-baseline.json"%self.base_path, history.history, to_float=True)


    def init_logits(self, refresh=False, train=True, test=True): # calculate logits from a certain layer in the classifier model
        if self.logits_train is None or refresh:
            if self.classifier_model is None:
                print("classifier model was None; loaded model from ", "%s/classifiermodel-weights-before.h5"%self.base_path, "instead")
                self.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-before.h5"%self.base_path)
#                 self.classifier_model = functions.get_model(self.model_name, optim = self.optim)
#                 self.classifier_model.load_weights("%s/classifiermodel-weights-before.h5"%self.base_path)
                
            # get outputs from a certain layer
            self.sub_model = tf.keras.Model(inputs=self.classifier_model.inputs, outputs=self.classifier_model.get_layer(self.layer_key).output)
            
            #print(self.X_train)
            if train:
                logits_train = self.sub_model.predict(self.X_train)
                self.logits_train = np.reshape(logits_train, (len(logits_train), -1)) # need to reshape to have only 2 dimensions
                self.embedding_config["embedding_size"] = self.logits_train.shape[1] #self.sub_model.output.shape[1]
                
            if test:
                logits_test = self.sub_model.predict(self.X_test)
                self.logits_test = np.reshape(logits_test, (len(logits_test), -1))
                self.embedding_config["embedding_size"] = self.logits_test.shape[1] #self.sub_model.output.shape[1]
            
        
        
        
    def get_shifted_data(self):
        self.init_logits()
            
        projected_train = self.embedder_model.encoder(self.logits_train)
#         projected_test = self.embedder_model.encoder(self.logits_test)
        
        labels_train = self.y_train
        shifted_train = projected_train.numpy()
        distinct_labels = list(set(labels_train))

        for k in self.contraction_factors:
            contraction_factor = self.contraction_factors[k]
            if k == "all":
                for i in range(len(distinct_labels)):
                    l = distinct_labels[i]
                    shifted_train[labels_train == l] = (contraction_factor[0] * shifted_train[(labels_train == l)]) + contraction_factor[1] * shifted_train[(labels_train == l)].mean(axis=0)
            else:
                shifted_train[labels_train == int(k)] = (contraction_factor[0] * shifted_train[(labels_train == k)]) + contraction_factor[1] * shifted_train[(labels_train == k)].mean(axis=0)
        
        for k in self.shift_factors:
            shift_factor = self.shift_factors[k]
            shifted_train[labels_train == int(k)] += np.array(shift_factor)
            
        return shifted_train
        
    def save_dict(self, file_path, data, to_float=False): # set to_float to true, if the dict consists of lists with float32 types that need to be converted to float64 types
        if to_float:
            for h in data:
                data[h] = list(np.array(data[h]).astype('float'))
        file = open(file_path, "w")
        json.dump(data, file)
        file.close()
        print("saved file to", file_path)
        
    def get_dict(self, file_path):
        file = open(file_path, "r")
        data = json.load(file)
        file.close()
        return data
    
    def get_histories(self):
        return self.get_dict("%s/classifiermodel-history-before.json"%self.base_path), self.get_dict("%s/classifiermodel-history-after-embedding.json"%self.base_path), self.get_dict("%s/classifiermodel-history-after-baseline.json"%self.base_path), {}#self.get_dict("%s/embeddermodel-history.json"%self.base_path)
        
    def load_models(self, load_train_logits=True, load_test_logits=True):
        self.classifier_model = functions.get_model(self.model_name, optim =  functions.get_optim(self.meta_data["optim"], self.meta_data["optim_config"]))
        self.classifier_model.load_weights("%s/classifiermodel-weights-before.h5"%self.base_path)
#         self.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-before.h5"%self.base_path)
        
        # get outputs from a certain layer
        #self.sub_model = tf.keras.Model(inputs=self.classifier_model.inputs, outputs=self.classifier_model.get_layer(self.layer_key).output)
        #self.embedding_config["embedding_size"] = self.sub_model.output.shape[1]
        self.init_logits(train=load_train_logits, test=load_test_logits)
        
        self.embedder_model = functions.get_model(self.embedding_approach, optim=self.embedding_optim, batch_size=self.embedding_batch_size, epochs=self.embedding_epochs, save_path="%s/embeddermodel"%self.base_path, config=self.embedding_config)
        
    
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plis = None

def get_plis(use_experiment_paths, load_models=False):
    global plis
    if plis is None or use_experiment_paths is not None: # refresh plis, if experiment paths are available
        plis = []
        for experiment_path in use_experiment_paths:
            pli = get_pli(experiment_path, load_models=False)
            plis.append(pli)
    return plis

def get_pli(experiment_path, load_models=False):
    with open("%smeta_data.json"%experiment_path) as json_file:
        meta_data = json.load(json_file)
        meta_data["base_path"] = experiment_path
        pli = Latent_Interventions(meta_data)
        if load_models:
            pli.load_models(load_train_logits=False)
        return pli

def plot_histories(use_experiment_paths, metric_key_base="accuracy", metric_key_embed="classifier_accuracy", name="", plot_baseline=False):
    plis = get_plis(use_experiment_paths)
        
    y_base = []
    y_embed = []
    x = []

    for i in range(len(plis)):
        pli = plis[i]
        history_model_before, history_after_embedding, history_after_baseline, history_embedder = pli.get_histories()
        
        y_base.extend(history_model_before[metric_key_base])
        y_embed.extend(history_model_before[metric_key_base])
        x.extend(list(range(1, pli.epochs+1)))
        
        y_base.extend(history_after_baseline[metric_key_base])
        y_embed.extend(history_after_embedding[metric_key_embed])
        x.extend(list(range(pli.epochs+1, pli.epochs+pli.post_epochs+1)))

    if plot_baseline:
        ax = sns.lineplot(x=x, y=y_base, label="Baseline %s - %s"%(metric_key_base, name), ci=95) #ci=95 -> 95% confidence interval
    ax = sns.lineplot(x=x, y=y_embed, label="Embedding %s - %s"%(metric_key_embed, name), ci=95)
    
    plt.xlabel("epoch")
    plt.ylabel(metric_key_base)
    
#     del plis
    
    
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

def plot_embedding(embedding, y_train, title=""):
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    colors = [plt.cm.tab10.colors[i] for i in y_train]
    ax.scatter(embedding[:,0], embedding[:,1], c=colors, s=2)
    ax.set_aspect(1)
    recs = []
    for i in range(0,10):
        recs.append(mpatches.Rectangle((0,0),1,1,fc=plt.cm.tab10.colors[i]))
    ax.legend(recs,list(range(10)),loc=2)
    plt.title(title)
    plt.show()
    
def plot_embeddings_from_pli(pli, name=""):
    print("train embedding of experiment nr. %s with %s"%(pli.meta_data["experiment_number"], name))
        
    # plot embedding of train data BEFORE 
    pli.init_logits()
    projected_train = pli.embedder_model.encoder(pli.logits_train)
    plot_embedding(projected_train, pli.y_train, title="embedding of train data")

    # plot embedding of the SHIFTED train data BEFORE
    plot_embedding(pli.get_shifted_data(), pli.y_train, title="embedding of shifted train data")

    # plot embedding of the train data AFTER it was trained as usual
    pli.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-after-baseline.h5"%pli.base_path)
    pli.init_logits(test=False)
    projected_train = pli.embedder_model.encoder(pli.logits_train)
    plot_embedding(projected_train, pli.y_train, title="embedding of train data - after-baseline")

    # plot embedding of the train data AFTER it was trained with the SHIFTED train data
    pli.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-after-embedding.h5"%pli.base_path)
    pli.init_logits(test=False)
    projected_train = pli.embedder_model.encoder(pli.logits_train)
    plot_embedding(projected_train, pli.y_train, title="embedding of train data - after-embedding")
    
def plot_embeddings_from_pli_list_train(use_experiment_paths, name=""):
    plis = get_plis(use_experiment_paths, load_models=True)
    
    for pli in plis:
        plot_embeddings_from_pli(pli, name)
        
#     del plis
        
def plot_embeddings_from_pli_list_test(use_experiment_paths, name=""):
    plis = get_plis(use_experiment_paths, load_models=True)
    
    for pli in plis:
        print("test embedding of experiment nr. %s with %s"%(pli.meta_data["experiment_number"], name))
        
        # plot embedding of the test data AFTER it was trained as usual
        pli.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-after-baseline.h5"%pli.base_path)
        pli.init_logits(train=False)
        projected_test = pli.embedder_model.encoder(pli.logits_test)
        plot_embedding(projected_test, pli.y_test, title="embedding of test data - after-baseline")
        
        # plot embedding of the test data AFTER it was trained with the SHIFTED train data
        pli.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-after-embedding.h5"%pli.base_path)
        pli.init_logits(train=False)
        projected_test = pli.embedder_model.encoder(pli.logits_test)
        plot_embedding(projected_test, pli.y_test, title="embedding of test data - after-embedding")
        
#     del plis
        
def test_set_evaluation(use_experiment_paths, name="", plot_cm=False):
    plis = get_plis(use_experiment_paths, load_models=True)
    base_accs = np.zeros(len(plis))
    emb_accs = np.zeros(len(plis))
    for j in range(len(plis)):
        pli = plis[j]
        print("\ntest-set evaluation of experiment nr. %s with %s"%(pli.meta_data["experiment_number"], name))
        
        n_classes = len(list(set(pli.y_test)))
        
        # test data AFTER it was trained as usual
        pli.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-after-baseline.h5"%pli.base_path)
        cf_test_baseline = np.zeros((n_classes, n_classes), dtype=int)
        for i in zip(pli.y_test, pli.classifier_model.predict(pli.X_test).argmax(1)):
            cf_test_baseline[i] += 1

        # test data AFTER it was trained with the SHIFTED train data
        pli.classifier_model = tf.keras.models.load_model("%s/classifiermodel-weights-after-embedding.h5"%pli.base_path)
        cf_test_altered = np.zeros((n_classes, n_classes), dtype=int)
        for i in zip(pli.y_test, pli.classifier_model.predict(pli.X_test)[0].argmax(1)): # this model has to outputs (classifier + embedder)
            cf_test_altered[i] += 1
            
        if plot_cm:
            plt.figure(figsize=(15,10))
            sns.heatmap(cf_test_baseline, annot=True)
            plt.title("confusion matrix of baseline")
            plt.show()
            plt.figure(figsize=(15,10))
            sns.heatmap(cf_test_altered, annot=True)
            plt.title("confusion matrix of intervention")
            plt.show()

        # baseline accuracy
        base_accs[j] = cf_test_baseline.diagonal().sum() / cf_test_baseline.sum()
        print("test accuracy of baseline:", cf_test_baseline.diagonal().sum() / cf_test_baseline.sum())

        # intervention accuracy
        emb_accs[j] = cf_test_altered.diagonal().sum() / cf_test_altered.sum()
        print("test accuracy of intervention:", cf_test_altered.diagonal().sum() / cf_test_altered.sum())
        
        accs = {"base_acc": base_accs[j], "emb_acc": emb_accs[j]}
        pli.save_dict(pli.base_path + "/testset-eval.json", accs, to_float=False)
        
#     del plis
    return base_accs, emb_accs
        
        
import functions.cluster_metrics as cm
def cluster_evaluation(use_experiment_paths):
    for experiment_path in use_experiment_paths:
        pli = get_pli(experiment_path, load_models=True)
        print(pli.base_path)
        metrics_dict_all = {}
        for model_path in ["classifiermodel-weights-before", "classifiermodel-weights-after-baseline", "classifiermodel-weights-after-embedding"]:
            print(model_path)
            print("---high dim space---")
            metrics_dict = {}
    
            pli.classifier_model = tf.keras.models.load_model("%s/%s.h5"%(experiment_path,model_path))
            pli.init_logits(refresh=True)
            
            print("train")
            #perform all cluster_metric algorithms here
            cluster_metrics = cm.ClusterMetrics(pli.logits_train, pli.y_train)
            metrics_dict["train_silhouette"] = cluster_metrics.get_silhouette_score()
            metrics_dict["train_davies_bouldin"] = cluster_metrics.get_davies_bouldin_score()
            # metrics_dict["train_dunn"] = cluster_metrics.get_dunn_score() # gets memory exception
            
            print("test")
            #perform all cluster_metric algorithms here
            cluster_metrics = cm.ClusterMetrics(pli.logits_test, pli.y_test)
            metrics_dict["test_silhouette"] = cluster_metrics.get_silhouette_score()
            metrics_dict["test_davies_bouldin"] = cluster_metrics.get_davies_bouldin_score()
            metrics_dict["test_dunn"] = cluster_metrics.get_dunn_score()
            

            print("---2D space---")
            print("train")
            #perform all cluster_metric algorithms here
            cluster_metrics = cm.ClusterMetrics(pli.embedder_model.encoder(pli.logits_train), pli.y_train)
            metrics_dict["train_silhouette_2D"] = cluster_metrics.get_silhouette_score()
            metrics_dict["train_davies_bouldin_2D"] = cluster_metrics.get_davies_bouldin_score()
#             metrics_dict["train_dunn"] = cluster_metrics.get_dunn_score() # gets memory exception
            
            print("test")
            #perform all cluster_metric algorithms here
            cluster_metrics = cm.ClusterMetrics(pli.embedder_model.encoder(pli.logits_test), pli.y_test)
            metrics_dict["test_silhouette_2D"] = cluster_metrics.get_silhouette_score()
            metrics_dict["test_davies_bouldin_2D"] = cluster_metrics.get_davies_bouldin_score()
            metrics_dict["test_dunn_2D"] = cluster_metrics.get_dunn_score()
            
            metrics_dict_all[model_path] = metrics_dict
            
        print(metrics_dict_all)
        pli.save_dict(pli.base_path + "/cluster-eval.json", metrics_dict_all, to_float=False)
    
    