"""uefla-ablation: A Flower / TensorFlow app."""

import os

import tensorflow as tf
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def load_model():
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    inputs = tf.keras.Input(shape=(10,))
    x = tf.keras.layers.Dense(5, activation='relu')(inputs)

    # Output 1: Fault Type
    #output1 = tf.keras.layers.Dense(4, activation='softmax')(x)
    output1 = tf.keras.layers.Dense(4, activation='softmax', name='fault_type')(x)

    # Output 2: Fault Detection
    output2 = tf.keras.layers.Dense(1, activation='sigmoid', name='fault_detection')(x)

    model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])

    model.compile(
        optimizer='adam',
        loss=['sparse_categorical_crossentropy', 'binary_crossentropy'],
        metrics=['accuracy', 'accuracy']
    )
    return model


fds = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):
    # Download and partition dataset
    # Only initialize `FederatedDataset` once
    global fds
    #if fds is None:
    #    partitioner = IidPartitioner(num_partitions=num_partitions)
    #    fds = FederatedDataset(
    #        dataset="uoft-cs/cifar10",
    #        partitioners={"train": partitioner},
    #    )
    #partition = fds.load_partition(partition_id, "train")
    #partition.set_format("numpy")

    # Divide data on each node: 80% train, 20% test
    #partition = partition.train_test_split(test_size=0.2)
    #x_train, y_train = partition["train"]["img"] / 255.0, partition["train"]["label"]
    #x_test, y_test = partition["test"]["img"] / 255.0, partition["test"]["label"]
    DATASET_PATH = '/Users/marcellomaugeri/Documents/PhD/Projects/User-Empowered-Federated-Learning-in-Automotive/EngineFaultDB_Final.csv'
    FEATURES_TO_REMOVE = ['Fault','CO', 'CO2', 'O2', 'HC']
    ds = pd.read_csv(DATASET_PATH)
    #ablation study: remove the features that can measured only with external instrumentation
    X = ds.drop(columns=FEATURES_TO_REMOVE)
    X = MinMaxScaler().fit_transform(X)
    Y = ds['Fault']
        
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=True) 
    #Create a new Y_binary for fault detection only
    Y_train_binary = (Y_train != 0).astype(int).values
    Y_test_binary = (Y_test != 0).astype(int).values
    return X_train, Y_train, Y_train_binary, X_test, Y_test, Y_test_binary
