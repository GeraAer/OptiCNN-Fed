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

# 加载数据集
data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')

# 数据清理函数
def clean_data(data):
    # 将 inf/-inf 值替换为 NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 处理异常 0 值
    columns_to_check = data.columns[:-1]  # 忽略最后一列（标签列）

    for column in columns_to_check:
        zero_count = (data[column] == 0).sum()
        total_count = len(data[column])
        zero_ratio = zero_count / total_count

        if zero_ratio <= 0.5:
            # 打印出要删除的行
            rows_to_delete = data[data[column] == 0]
            print(f"列 {column} 中的 0 值占比较低，删除以下行：\n{rows_to_delete}")
            # 删除包含 0 值的行
            data = data[data[column] != 0]

    return data

# 对 data1 和 data2 进行清理
data1 = clean_data(data1)
data2 = clean_data(data2)

# 分离特征和标签
features1 = data1.iloc[:, :-1]
labels1 = data1.iloc[:, -1]
features2 = data2.iloc[:, :-1]
labels2 = data2.iloc[:, -1]

# 对特征进行缺失值填充和标准化
imputer = SimpleImputer(strategy='mean')
scaler = StandardScaler()

# 训练集处理
features1_imputed = pd.DataFrame(imputer.fit_transform(features1), columns=features1.columns)
features1_scaled = scaler.fit_transform(features1_imputed)

# 测试集处理
features2_imputed = pd.DataFrame(imputer.transform(features2), columns=features2.columns)
features2_scaled = scaler.transform(features2_imputed)

# 标签二值化
label_encoder = LabelEncoder()
labels1_encoded = label_encoder.fit_transform(labels1)
labels2_encoded = label_encoder.fit_transform(labels2)

# 组合处理后的特征和标签
train_data_processed = pd.DataFrame(features1_scaled, columns=features1.columns)
train_data_processed['Label'] = labels1_encoded

test_data_processed = pd.DataFrame(features2_scaled, columns=features2.columns)
test_data_processed['Label'] = labels2_encoded

# 将处理好的训练集和测试集保存为新文件
train_data_processed.to_csv('processed_training.csv', index=False)
test_data_processed.to_csv('processed_testing.csv', index=False)