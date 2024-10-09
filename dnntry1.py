import collections
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tensorflow_federated as tff
import csv
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

np.random.seed(1337)
import nest_asyncio

nest_asyncio.apply()

# Load the dataset
data = pd.read_csv('data1.csv')

# Data Cleaning
# Replace inf/-inf values with NaN
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Separate features and labels
features = data.iloc[:, :-1]
labels = data.iloc[:, -1]

# Impute missing values using the mean strategy for features
imputer = SimpleImputer(strategy='mean')
features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)

# Convert labels from 'natural'/'attack' to binary (0 and 1)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Standardize features (Z-score standardization)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_imputed)

# Combine scaled features with the label
data_processed = pd.DataFrame(features_scaled, columns=features.columns)
data_processed['Label'] = labels_encoded

# Split the dataset into training and testing sets
traindata = data_processed.sample(frac=0.8, random_state=42)
testdata = data_processed.drop(traindata.index)

# Reset index
traindata = traindata.reset_index(drop=True)
testdata = testdata.reset_index(drop=True)

# Define constants
CLIENTS_NUM = 5
ROUND_NUM = 200
client_lr = 0.02
server_lr = 1.0
batchSize = 64

# Split training data among clients
split_train_data = []
remain_traindata = traindata
for i in range(CLIENTS_NUM):
    temp_split_train_data = traindata.sample(frac=1 / CLIENTS_NUM)
    temp_split_train_data = temp_split_train_data.reset_index(drop=True)
    split_train_data.append(temp_split_train_data)
    remain_traindata = traindata[~traindata.index.isin(temp_split_train_data.index)]

# Split testing data among clients
split_test_data = []
remain_testdata = testdata
for i in range(CLIENTS_NUM):
    temp_split_test_data = testdata.sample(frac=1 / CLIENTS_NUM)
    temp_split_test_data = temp_split_test_data.reset_index(drop=True)
    split_test_data.append(temp_split_test_data)
    remain_testdata = testdata[~testdata.index.isin(temp_split_test_data.index)]

# Map clients to datasets
client_ids = [x for x in range(CLIENTS_NUM)]
train_ds = {}
for i in range(CLIENTS_NUM):
    cur_traindata = split_train_data[i]
    train_ds[i] = tf.data.Dataset.from_tensor_slices(cur_traindata)

test_ds = {}
for i in range(CLIENTS_NUM):
    cur_testdata = split_test_data[i]
    test_ds[i] = tf.data.Dataset.from_tensor_slices(cur_testdata)


def create_tf_traindataset_for_client_fn(i):
    return train_ds[i]


def create_tf_testdataset_for_client_fn(i):
    return test_ds[i]


# Create ClientData objects
TrainData4Client = tff.simulation.ClientData.from_clients_and_fn(client_ids, create_tf_traindataset_for_client_fn)
TestData4Client = tff.simulation.ClientData.from_clients_and_fn(client_ids, create_tf_testdataset_for_client_fn)

# Sample clients
sample_clients = TrainData4Client.client_ids[0:CLIENTS_NUM]


# Preprocess dataset
def preprocess_dataset(dataset):
    def map_fn(input):
        return collections.OrderedDict(
            x=tf.reshape(input[:, :-1], [-1, 128]),  # Adjusted to 128 features
            y=tf.reshape(input[:, -1], [-1, 1])
        )

    return dataset.batch(batchSize).map(map_fn).take(120000)


# Generate federated datasets
federated_train_datasets = [preprocess_dataset(TrainData4Client.create_tf_dataset_for_client(x)) for x in
                            sample_clients]
federated_test_datasets = [preprocess_dataset(TestData4Client.create_tf_dataset_for_client(x)) for x in sample_clients]

input_spec = federated_train_datasets[0].element_spec


# Federated Learning Model
def create_keras_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_dim=128, activation=tf.nn.relu))  # Adjusted input_dim to 128
    model.add(tf.keras.layers.Dropout(0.01))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation('sigmoid'))
    return model


def model_fn():
    model = create_keras_model()
    return tff.learning.from_keras_model(
        keras_model=model,
        input_spec=input_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.Accuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
    )


fed_aver = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=client_lr),
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=server_lr)
)

state = fed_aver.initialize()

logdir_for_compression = "/tmp/logs/scalars/custom_model/"
summary_writer = tf.summary.create_file_writer(logdir_for_compression)

f1 = open('evaluation.csv', 'w', encoding='utf-8')
f2 = open('test_evaluation.csv', 'w', encoding='utf-8')
csv_writer_train = csv.writer(f1)
csv_writer_test = csv.writer(f2)

csv_writer_train.writerow(["loss", "accuracy", "recall", "precision", "f1-score"])
csv_writer_test.writerow(["loss", "accuracy", "recall", "precision", "f1-score"])

with summary_writer.as_default():
    # Training loop
    for i in range(ROUND_NUM):
        state, metrics = fed_aver.next(state, federated_train_datasets)
        test_state, test_metrics = fed_aver.next(state, federated_test_datasets)
        train_f1_score = 2 * metrics['train']['recall'] * metrics['train']['precision'] / (
                    metrics['train']['recall'] + metrics['train']['precision'])
        test_f1_score = 2 * test_metrics['train']['recall'] * test_metrics['train']['precision'] / (
                    test_metrics['train']['recall'] + test_metrics['train']['precision'])

        print(
            f'Round {i}: Train Loss: {metrics["train"]["loss"]}, Accuracy: {metrics["train"]["accuracy"]}, Recall: {metrics["train"]["recall"]}, Precision: {metrics["train"]["precision"]}, F1-score: {train_f1_score}\n')
        csv_writer_train.writerow([metrics['train']['loss'], metrics['train']['accuracy'], metrics['train']['recall'],
                                   metrics['train']['precision'], train_f1_score])

        print(
            f'Round {i}: Test Loss: {test_metrics["train"]["loss"]}, Accuracy: {test_metrics["train"]["accuracy"]}, Recall: {test_metrics["train"]["recall"]}, Precision: {test_metrics["train"]["precision"]}, F1-score: {test_f1_score}\n')
        csv_writer_test.writerow(
            [test_metrics['train']['loss'], test_metrics['train']['accuracy'], test_metrics['train']['recall'],
             test_metrics['train']['precision'], test_f1_score])

f1.close()
f2.close()