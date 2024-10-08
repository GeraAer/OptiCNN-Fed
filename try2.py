import collections
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
import os
import asyncio
import random
from deap import base, creator, tools, algorithms
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold

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
# Replace inf values and fill NaN values to avoid large or invalid numbers
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

# Genetic Algorithm for Neural Network Architecture Search
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define individual generation function
def generate_individual():
    num_layers = random.randint(1, 5)  # Number of hidden layers (1 to 5)
    neurons_per_layer = [random.randint(16, 1024) for _ in range(num_layers)]  # Neurons per layer (16 to 1024)
    return [num_layers] + neurons_per_layer

# Register functions to create population
toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual, generate_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate_nn(individual):
    # Create model based on individual's genome (number of layers and neurons per layer)
    layers = []
    input_shape = (features.shape[1],)

    for i in range(individual[0]):  # The first gene is the number of hidden layers
        layers.append(tf.keras.layers.Dense(individual[i + 1], activation='relu'))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dropout(0.3))

    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_shape)] + layers)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Split the dataset into training and testing sets (80% training, 20% testing)
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    train_index, test_index = next(kf.split(features))
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = labels[train_index], labels[test_index]

    # Train model and return accuracy as fitness
    model.fit(X_train, y_train, epochs=5, verbose=0)
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    return accuracy,

# Register the evaluation, selection, crossover, and mutation functions
toolbox.register("evaluate", evaluate_nn)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt, low=16, up=1024, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Create initial population
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
population = toolbox.population(n=POPULATION_SIZE)

# Run the genetic algorithm
for gen in range(NUM_GENERATIONS):
    print(f"Generation {gen}")

    # Evaluate individuals
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Select and generate offspring
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # Apply crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < 0.5:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < 0.2:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Replace population with the new offspring
    population[:] = offspring

# Print best individual
best_individual = tools.selBest(population, 1)[0]
print(f"Best individual: {best_individual}, Fitness: {best_individual.fitness.values}")

# Use the best individual to define the final model
def create_keras_model():
    layers = []
    input_shape = (features.shape[1],)

    for i in range(best_individual[0]):
        layers.append(tf.keras.layers.Dense(best_individual[i + 1], activation='relu'))
        layers.append(tf.keras.layers.BatchNormalization())
        layers.append(tf.keras.layers.Dropout(0.3))

    layers.append(tf.keras.layers.Dense(1, activation='sigmoid'))

    model = tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_shape)] + layers)
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
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=0.002),  # Reduce learning rate for finer updates
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.1)  # Reduce server learning rate to prevent instability
)

# Initialize the federated learning process
state = fed_avg.initialize()

# Perform federated learning using the best model
NUM_ROUNDS = 30
kf = KFold(n_splits=10, shuffle=True, random_state=42)
fold = 1
all_metrics = []

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
        )).batch(16)
        train_ds.append(dataset)

    # Prepare test dataset
    test_data = tf.data.Dataset.from_tensor_slices((
        X_test.astype(np.float32),
        y_test.astype(np.float32)
    )).batch(16)

    # Prepare federated dataset
    client_ids = list(range(CLIENTS_NUM))
    federated_train_data = [preprocess(dataset) for dataset in train_ds]

    # Training loop
    for round_num in range(NUM_ROUNDS):
        state, metrics = fed_avg.next(state, federated_train_data)
        print(f'Fold {fold}, Round {round_num + 1}, Metrics: {metrics}')

    # Evaluate the model on the test data
    final_model = model_fn()
    state.model.assign_weights_to(final_model)
    test_metrics = final_model.keras_model.evaluate(test_data, return_dict=True)
    print(f'Fold {fold}, Test Metrics:', test_metrics)
    all_metrics.append(test_metrics)
    fold += 1

# Print average metrics across all folds
average_metrics = {key: np.mean([m[key] for m in all_metrics]) for key in all_metrics[0].keys()}
print('Average Test Metrics:', average_metrics)