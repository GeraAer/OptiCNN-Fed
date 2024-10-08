import collections
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tensorflow_federated as tff
import csv
import nest_asyncio

# Setting GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

np.random.seed(1337)
nest_asyncio.apply()

# Load training and testing datasets
traindata = pd.read_csv('kdd/binary/Training.csv', header=0)
testdata = pd.read_csv('kdd/binary/Testing.csv', header=0)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for column in traindata.columns:
    if traindata[column].dtype == 'object':
        traindata[column] = le.fit_transform(traindata[column])
for column in testdata.columns:
    if testdata[column].dtype == 'object':
        testdata[column] = le.fit_transform(testdata[column])

# Set the number of clients
CLIENTS_NUM = 5

# Split training data among clients
split_train_data = []
for i in range(CLIENTS_NUM):
    temp_split_train_data = traindata.sample(frac=1 / CLIENTS_NUM).reset_index(drop=True)
    split_train_data.append(temp_split_train_data)

# Split testing data among clients
split_test_data = []
for i in range(CLIENTS_NUM):
    temp_split_test_data = testdata.sample(frac=1 / CLIENTS_NUM).reset_index(drop=True)
    split_test_data.append(temp_split_test_data)

# Create TensorFlow datasets for each client
train_ds = {}
test_ds = {}
for i in range(CLIENTS_NUM):
    train_ds[i] = tf.data.Dataset.from_tensor_slices(split_train_data[i].values.astype(np.float32)).batch(64)
    test_ds[i] = tf.data.Dataset.from_tensor_slices(split_test_data[i].values.astype(np.float32)).batch(64)

# Client data function
def create_tf_dataset_for_client_fn(client_id):
    return train_ds[client_id]

def create_tf_test_dataset_for_client_fn(client_id):
    return test_ds[client_id]

client_ids = [x for x in range(CLIENTS_NUM)]
TrainData4Client = tff.simulation.ClientData.from_clients_and_fn(client_ids, create_tf_dataset_for_client_fn)
TestData4Client = tff.simulation.ClientData.from_clients_and_fn(client_ids, create_tf_test_dataset_for_client_fn)

# Preprocess dataset
def preprocess(dataset):
    def map_fn(input):
        return collections.OrderedDict(
            x=input[1:],
            y=input[0]
        )
    return dataset.map(map_fn)

# Prepare federated dataset
federated_train_data = [preprocess(TrainData4Client.create_tf_dataset_for_client(x)) for x in client_ids]
federated_test_data = [preprocess(TestData4Client.create_tf_dataset_for_client(x)) for x in client_ids]

# Create Keras model
def create_keras_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(traindata.shape[1] - 1,)),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Define TFF model
def model_fn():
    keras_model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model,
        input_spec=federated_train_data[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(),
            tf.keras.metrics.Precision(),
            tf.keras.metrics.Recall()
        ]
    )

# Federated Averaging process definition
fed_avg = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0)
)

# Initialize the federated learning process
state = fed_avg.initialize()

# Training loop
NUM_ROUNDS = 200
logdir = '/tmp/logs/scalars/custom_model/'
summary_writer = tf.summary.create_file_writer(logdir)

with summary_writer.as_default():
    for round_num in range(NUM_ROUNDS):
        state, metrics = fed_avg.next(state, federated_train_data)
        print(f'Round {round_num + 1}: {metrics}')
        tf.summary.scalar('loss', metrics['train']['loss'], step=round_num)
        tf.summary.scalar('accuracy', metrics['train']['binary_accuracy'], step=round_num)
        tf.summary.scalar('precision', metrics['train']['precision'], step=round_num)
        tf.summary.scalar('recall', metrics['train']['recall'], step=round_num)

# Evaluation loop
print("Final Training Metrics:")
print(metrics)

# Evaluate on test data
for client_id in client_ids:
    test_metrics = model_fn().evaluate(federated_test_data[client_id], return_dict=True)
    print(f'Test metrics for client {client_id}: {test_metrics}')