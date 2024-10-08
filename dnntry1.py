import collections
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
import os
import asyncio
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
import random

# Setting GPU environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'

# Ensure event loop is set properly
nest_asyncio.apply()
try:
    asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Load the dataset 'data1'
data = pd.read_csv('data1.csv', header=0)

# Extract features and labels
features = data.iloc[:, :-1]
labels = data['marker']

# Encode the labels into binary values (Natural: 0, Attack: 1)
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Clip labels to ensure they are strictly binary (0 or 1)
labels = np.clip(labels, 0, 1)

# Ensure all data is in numeric format
def preprocess_features(df):
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = LabelEncoder().fit_transform(df[column])
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df

features = preprocess_features(features)

# Normalize the features using StandardScaler (Z-score normalization)
scaler = StandardScaler()
features = scaler.fit_transform(features)

# Set the number of clients
CLIENTS_NUM = 5

# Perform tenfold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold = 1
all_metrics = []

# Genetic Algorithm parameters
POPULATION_SIZE = 30  # Population size
GENERATIONS = 10  # Number of generations
CROSSOVER_RATE = 0.8  # Crossover probability
MUTATION_RATE = 0.2  # Mutation probability

# Define the model creation using parameters
class GeneticCNN:
    def __init__(self, num_features):
        self.num_features = num_features
        self.population = self.initialise_population()

    def initialise_population(self):
        population = []
        for _ in range(POPULATION_SIZE):
            individual = {
                'layers': random.randint(2, 5),  # Number of convolutional blocks
                'filters': [random.randint(16, 64) for _ in range(5)],  # Number of filters per block
                'kernel_size': [random.choice([3, 5]) for _ in range(5)],  # Kernel size per block
                'activation': random.choice(['relu', 'tanh']),
                'pool_size': random.choice([2, 3])  # Pooling size
            }
            population.append(individual)
        return population

    def create_keras_model(self, individual):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(self.num_features,)))
        model.add(tf.keras.layers.Reshape((self.num_features, 1)))  # Reshape to add an extra dimension for Conv1D
        for i in range(individual['layers']):
            model.add(tf.keras.layers.Conv1D(individual['filters'][i], individual['kernel_size'][i], activation=individual['activation'], padding='same'))
            pool_size = min(individual['pool_size'], model.output_shape[1])
            model.add(tf.keras.layers.MaxPooling1D(pool_size=pool_size))
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        return model

    def fitness(self, individual, X_train, y_train, X_val, y_val):
        model = self.create_keras_model(individual)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        _, accuracy = model.evaluate(X_val, y_val, verbose=0)
        return accuracy

    def select_parents(self, population, fitness_scores):
        selected = random.sample(list(zip(population, fitness_scores)), 5)  # 5-way tournament selection
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[0][0], selected[1][0]

    def crossover(self, parent1, parent2):
        child1, child2 = parent1.copy(), parent2.copy()
        if random.random() < CROSSOVER_RATE:
            crossover_point = random.randint(0, len(parent1['filters']) - 1)
            for key in parent1.keys():
                if isinstance(parent1[key], list):
                    child1[key][:crossover_point], child2[key][:crossover_point] = parent2[key][:crossover_point], parent1[key][:crossover_point]
                else:
                    child1[key], child2[key] = (parent1[key], parent2[key]) if random.random() < 0.5 else (parent2[key], parent1[key])
        return child1, child2

    def mutate(self, individual):
        if random.random() < MUTATION_RATE:
            mutation_point = random.choice(list(individual.keys()))
            if mutation_point == 'layers':
                individual['layers'] = random.randint(2, 5)
            elif mutation_point == 'filters':
                individual['filters'] = [random.randint(16, 64) for _ in range(5)]
            elif mutation_point == 'kernel_size':
                individual['kernel_size'] = [random.choice([3, 5]) for _ in range(5)]
            elif mutation_point == 'activation':
                individual['activation'] = random.choice(['relu', 'tanh'])
            elif mutation_point == 'pool_size':
                individual['pool_size'] = random.choice([2, 3])
        return individual

    def evolve_population(self, X_train, y_train, X_val, y_val):
        for generation in range(GENERATIONS):
            fitness_scores = [self.fitness(ind, X_train, y_train, X_val, y_val) for ind in self.population]
            new_population = []
            while len(new_population) < POPULATION_SIZE:
                parent1, parent2 = self.select_parents(self.population, fitness_scores)
                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)
                new_population.extend([child1, child2])
            self.population = new_population[:POPULATION_SIZE]
            print(f'Generation {generation + 1} best accuracy: {max(fitness_scores)}')

# Preprocess function for federated dataset
def preprocess(dataset):
    def map_fn(input, label):
        return collections.OrderedDict(
            x=input,
            y=label
        )
    return dataset.map(map_fn)

# Split the dataset into training and testing sets
for train_index, test_index in kf.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Split training data among clients
    train_data = pd.concat([pd.DataFrame(X_train), pd.Series(y_train, name='label')], axis=1)
    split_train_data = np.array_split(train_data, CLIENTS_NUM)

    # Create TensorFlow datasets for each client
    train_ds = []
    for client_data in split_train_data:
        dataset = tf.data.Dataset.from_tensor_slices((
            client_data.iloc[:, :-1].values.astype(np.float32),  # Features
            client_data['label'].values.astype(np.float32)       # Labels
        )).batch(32)
        train_ds.append(dataset)

    # Prepare test dataset
    test_data = tf.data.Dataset.from_tensor_slices((
        X_test.astype(np.float32),
        y_test.astype(np.float32)
    )).batch(32)

    # Prepare federated dataset
    client_ids = list(range(CLIENTS_NUM))
    federated_train_data = [preprocess(dataset) for dataset in train_ds]

    genetic_cnn = GeneticCNN(num_features=X_train.shape[1])
    genetic_cnn.evolve_population(X_train, y_train, X_test, y_test)

    # Train the best model
    best_model_params = genetic_cnn.population[0]
    final_model = genetic_cnn.create_keras_model(best_model_params)

    # Define TFF model
    def model_fn():
        return tff.learning.from_keras_model(
            final_model,
            input_spec=federated_train_data[0].element_spec,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.AUC(name='auc')
            ]
        )

    # Federated Averaging process definition
    fed_avg = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.0005),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.001)
    )

    # Initialize the federated learning process
    state = fed_avg.initialize()

    # Training loop
    NUM_ROUNDS = 200
    for round_num in range(NUM_ROUNDS):
        state, metrics = fed_avg.next(state, federated_train_data)
        print(f'Fold {fold}, Round {round_num + 1}, Metrics: {metrics}')

    # Evaluate the model on the test data
    state.model.assign_weights_to(final_model)

    # Evaluate on test data
    test_metrics = final_model.evaluate(test_data, return_dict=True)
    print(f'Fold {fold}, Test Metrics:', test_metrics)
    all_metrics.append(test_metrics)
    fold += 1

# Print average metrics across all folds
average_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
print('Average Test Metrics:', average_metrics)

# Calculate F1-score
average_precision = average_metrics['precision']
average_recall = average_metrics['recall']
average_f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall)
print('Average F1 Score:', average_f1_score)