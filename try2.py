import collections
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tensorflow_federated as tff
import random
from sklearn.preprocessing import LabelEncoder, StandardScaler
import nest_asyncio

# Setting up GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

np.random.seed(1337)
nest_asyncio.apply()

# Reading dataset and skipping the feature names
data = pd.read_csv('data1.csv', header=0)

# Extract features and labels
features = data.iloc[:, :-1].values  # All columns except the last one are features
labels = data['marker'].values  # Last column is the label

# Encode labels (Natural -> 0, Attack -> 1)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)


# Check and replace inf and NaN values
def replace_invalid_values(data):
    data = np.where(np.isinf(data), np.nan, data)  # Replace inf with NaN
    data = np.nan_to_num(data, nan=0.0)  # Replace NaN with 0
    return data


features = replace_invalid_values(features)

# Standardize features
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Number of clients
CLIENTS_NUM = 5

# Split data into multiple client datasets
split_data = np.array_split(np.column_stack((features, labels)), CLIENTS_NUM)

# Create TensorFlow datasets
train_ds = {}
for i in range(CLIENTS_NUM):
    split_features = split_data[i][:, :-1].astype(np.float32)
    split_labels = split_data[i][:, -1].astype(np.float32)
    train_ds[i] = tf.data.Dataset.from_tensor_slices((split_features, split_labels)).batch(64)


# Client data function
def create_tf_dataset_for_client_fn(client_id):
    return train_ds[client_id]


client_ids = [x for x in range(CLIENTS_NUM)]
TrainData4Client = tff.simulation.ClientData.from_clients_and_fn(client_ids, create_tf_dataset_for_client_fn)


# Preprocess dataset
def preprocess(dataset):
    def map_fn(features, label):
        return collections.OrderedDict(
            x=features,  # Input features
            y=label  # Label
        )

    return dataset.map(map_fn)


# Prepare federated training data
federated_train_data = [preprocess(TrainData4Client.create_tf_dataset_for_client(x)) for x in client_ids]

# Genetic Algorithm parameters
POPULATION_SIZE = 30
GENERATIONS = 10
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
LEARNING_RATE_RANGE = (0.0001, 0.01)


# Genetic Algorithm class to evolve CNN and learning rates
class GeneticFederatedCNN:
    def __init__(self, num_features):
        self.num_features = num_features
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(POPULATION_SIZE):
            individual = {
                'layers': random.randint(2, 5),  # Number of layers
                'filters': [random.randint(16, 64) for _ in range(5)],  # Filters for each layer
                'kernel_size': [random.choice([3, 5]) for _ in range(5)],  # Kernel size
                'activation': random.choice(['relu', 'tanh']),
                'client_lr': random.uniform(*LEARNING_RATE_RANGE),  # Client learning rate
                'server_lr': random.uniform(*LEARNING_RATE_RANGE)  # Server learning rate
            }
            population.append(individual)
        return population

    def create_keras_model(self, individual):
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(self.num_features,)),
            tf.keras.layers.Dense(1024, activation=individual['activation']),
            tf.keras.layers.Dropout(0.01),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
        ])
        return model

    def fitness(self, individual, federated_data):
        model = self.create_keras_model(individual)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=individual['client_lr']),
                      loss='binary_crossentropy', metrics=['accuracy'])

        # Simulate federated training with current individual's parameters
        fed_avg = tff.learning.build_federated_averaging_process(
            lambda: tff.learning.from_keras_model(
                model,
                input_spec=federated_data[0].element_spec,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()]),
            client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=individual['client_lr']),
            server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=individual['server_lr'])
        )

        state = fed_avg.initialize()
        for round_num in range(5):  # Simulate training for 5 rounds
            state, metrics = fed_avg.next(state, federated_data)
        return metrics['train']['binary_accuracy']  # Fitness based on accuracy

    def evolve_population(self):
        for generation in range(GENERATIONS):
            fitness_scores = [self.fitness(ind, federated_train_data) for ind in self.population]
            # Apply crossover, mutation, and selection...
            print(f'Generation {generation + 1}: Best fitness: {max(fitness_scores)}')


# Initialize and run the GA for federated CNN optimization
genetic_cnn = GeneticFederatedCNN(num_features=features.shape[1])
genetic_cnn.evolve_population()
