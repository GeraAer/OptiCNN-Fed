import collections
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import tensorflow_federated as tff
import csv

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'  # 这是默认的显示等级，忽略所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='2' # 忽略 warning 和 Error
# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 忽略 Error

np.random.seed(1337)
import nest_asyncio

nest_asyncio.apply()

traindata = pd.read_csv('kdd/binary/Training.csv', header=None)
testdata = pd.read_csv('kdd/binary/Testing.csv', header=None)
# 客户端个数
CLIENTS_NUM = 5
ROUND_NUM = 200
# 客户端学习率
client_lr = 0.02
# 服务端学习率
server_lr = 1.0
batchSize = 64

# 数据集划分，用于之后分配到每一个client上
split_train_data = []
remain_traindata = traindata
for i in range(5):
    temp_split_train_data = traindata.sample(frac=1 / CLIENTS_NUM)
    temp_split_train_data = temp_split_train_data.reset_index(drop=True)
    split_train_data.append(temp_split_train_data)
    remain_traindata = traindata[~traindata.index.isin(traindata[i].index)]

# 分割测试集数据
split_test_data = []
remain_testdata = testdata
for i in range(CLIENTS_NUM):
    temp_split_test_data = testdata.sample(frac=1 / CLIENTS_NUM)
    temp_split_test_data = temp_split_test_data.reset_index(drop=True)
    split_test_data.append(temp_split_test_data)
    remain_testdata = testdata[~testdata.index.isin(testdata[i].index)]

# 客户端与数据集对应
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


# 将结构化数据转化为clientdata类型的数据
TrainData4Client = tff.simulation.ClientData.from_clients_and_fn(client_ids, create_tf_traindataset_for_client_fn)
TestData4Client = tff.simulation.ClientData.from_clients_and_fn(client_ids, create_tf_testdataset_for_client_fn)

# 生成固定数量的客户端
sample_clients = TrainData4Client.client_ids[0:CLIENTS_NUM]


# 数据预处理
def preprocess_dataset(dataset):
    def map_fn(input):
        return collections.OrderedDict(x=tf.reshape(input[:, 1:], [-1, 41]), y=tf.reshape(input[:, 0], [-1, 1]))

    return dataset.batch(batchSize).map(map_fn).take(120000)


# 为每个客户端生成训练数据集
federated_train_datasets = [preprocess_dataset(TrainData4Client.create_tf_dataset_for_client(x)) for x in sample_clients]
federated_test_datasets = [preprocess_dataset(TestData4Client.create_tf_dataset_for_client(x)) for x in sample_clients]

input_spec = federated_train_datasets[0].element_spec


# 联邦学习模型
# dnn1
def create_keras_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(1024, input_dim=41, activation=tf.nn.relu))
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
    # 客户端优化器，只针对客户端本地模型进行更新优化
    client_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=client_lr),
    # 服务器端优化器，只针对服务器端全局模型进行更新优化
    server_optimizer_fn=lambda: tf.keras.optimizers.Adam(learning_rate=server_lr)
)

state = fed_aver.initialize()

logdir_for_compression = "/tmp/logs/scalars/custom_model/"
summary_writer = tf.summary.create_file_writer(
    logdir_for_compression)

f1 = open('evaluation.csv', 'w', encoding='utf-8')
f2 = open('test_evaluation.csv', 'w', encoding='utf-8')
csv_writer_train = csv.writer(f1)
csv_writer_test = csv.writer(f2)

csv_writer_train.writerow(["loss", "accuracy", "recall", "precision", "f1-score"])
csv_writer_test.writerow(["loss", "accuracy", "recall", "precision", "f1-score"])


with summary_writer.as_default():
    # 基础训练测试
    for i in range(ROUND_NUM):
        state, metrics = fed_aver.next(state, federated_train_datasets)
        test_state, test_metrics = fed_aver.next(state, federated_test_datasets)
        print('第', i , '轮训练模型loss：',  metrics['train']['loss'] , '准确率：', metrics['train']['accuracy'] ,'召回率：', metrics['train']['recall'],'精度：', metrics['train']['precision'] ,'f1-score：',(2*metrics['train']['recall']* metrics['train']['precision']/(metrics['train']['recall']+metrics['train']['precision'])), '\n')
        csv_writer_train.writerow(
            [metrics['train']['loss'], metrics['train']['accuracy'], metrics['train']['recall'],
             metrics['train']['precision'], (
                     2 * metrics['train']['recall'] * metrics['train']['precision'] / (
                     metrics['train']['recall'] + metrics['train']['precision']))])
        print('第', i , '轮测试模型loss：',  metrics['train']['loss'] , '准确率：', metrics['train']['accuracy'] ,'召回率：', metrics['train']['recall'],'精度：', metrics['train']['precision'] ,'f1-score：',(2*metrics['train']['recall']* metrics['train']['precision']/(metrics['train']['recall']+metrics['train']['precision'])), '\n')
        csv_writer_test.writerow(
            [test_metrics['train']['loss'], test_metrics['train']['accuracy'], test_metrics['train']['recall'],
             test_metrics['train']['precision'], (
                     2 * test_metrics['train']['recall'] * test_metrics['train']['precision'] / (
                     test_metrics['train']['recall'] + test_metrics['train']['precision']))])

