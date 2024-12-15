import tensorflow as tf
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

SAVED_MODEL_DIR = "saved_model"
DATASET_PATH = '../EngineFaultDB_Final.csv'
FEATURES_TO_REMOVE = ['Fault','CO', 'CO2', 'O2', 'HC']

def load_data():
    ds = pd.read_csv(DATASET_PATH)
    #ablation study: remove the features that can measured only with external instrumentation
    X = ds.drop(columns=FEATURES_TO_REMOVE)
    X = MinMaxScaler().fit_transform(X)
    Y = ds['Fault']
        
    return train_test_split(X, Y, test_size=0.2, shuffle=True) 

X_train, X_test, Y_train, Y_test = load_data()


#Create a new Y_binary for fault detection only
Y_train_binary = (Y_train != 0).astype(int).values
Y_test_binary = (Y_test != 0).astype(int).values

#Print shapes
print(f"X_train shape: {X_train.shape}")
print(f"Y_train shape: {Y_train.shape}")
print(f"Y_train_binary shape: {Y_train_binary.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Y_test shape: {Y_test.shape}")
print(f"Y_test_binary shape: {Y_test_binary.shape}")

print("Y_train_binary sample:", Y_train_binary[:5])

# Functional API approach
inputs = tf.keras.Input(shape=(X_train.shape[1],))
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

# Check the model summary to ensure outputs are connected correctly
model.summary()


model.fit(X_train, [Y_train, Y_train_binary ], epochs=30, batch_size=16)
model.evaluate(X_test, [Y_test, Y_test_binary])
#print("Saving model")
#model.save('enginefaultdb.keras')