import pandas as pd  # 导入pandas库，用于数据处理
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score)  # 从sklearn库中导入度量函数

# 从文件中读取训练数据和测试数据
traindata = pd.read_csv('expected.txt', header=None)  # 读取训练数据，即实际标签
testdata = pd.read_csv('predictedlabelAB.txt', header=None)  # 读取AB模型的预测标签

# 定义标签变量
y_train1 = traindata  # 真实标签
y_pred = testdata  # 预测标签

# 计算AB模型的各项评估指标
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1值

# 打印AB模型的结果
print("ABresults")
print("accuracy")  # 打印准确率
print("%.3f" % accuracy)  # 保留三位小数的准确率
print("precision")  # 打印精确率
print("%.3f" % precision)  # 保留三位小数的精确率
print("racall")  # 打印召回率（这里有个拼写错误，应该是recall）
print("%.3f" % recall)  # 保留三位小数的召回率
print("f1score")  # 打印F1值
print("%.3f" % f1)  # 保留三位小数的F1值

# 读取其他模型的预测标签并重复上述过程

testdata = pd.read_csv('predictedlabelDT.txt', header=None)  # 读取DT模型的预测标签
y_train1 = traindata  # 真实标签
y_pred = testdata  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1值

# 打印DT模型的结果
print("DTresults")
print("accuracy")  # 打印准确率
print("%.3f" % accuracy)  # 保留三位小数的准确率
print("precision")  # 打印精确率
print("%.3f" % precision)  # 保留三位小数的精确率
print("racall")  # 打印召回率（拼写错误，应该是recall）
print("%.3f" % recall)  # 保留三位小数的召回率
print("f1score")  # 打印F1值
print("%.3f" % f1)  # 保留三位小数的F1值

# 读取KNN模型的预测标签并重复上述过程
testdata = pd.read_csv('predictedlabelKNN.txt', header=None)  # 读取KNN模型的预测标签
y_train1 = traindata  # 真实标签
y_pred = testdata  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1值

# 打印KNN模型的结果
print("KNNresults")
print("accuracy")  # 打印准确率
print("%.3f" % accuracy)  # 保留三位小数的准确率
print("precision")  # 打印精确率
print("%.3f" % precision)  # 保留三位小数的精确率
print("racall")  # 打印召回率（拼写错误，应该是recall）
print("%.3f" % recall)  # 保留三位小数的召回率
print("f1score")  # 打印F1值
print("%.3f" % f1)  # 保留三位小数的F1值

# 读取NB模型的预测标签并重复上述过程
testdata = pd.read_csv('predictedlabelNB.txt', header=None)  # 读取NB模型的预测标签
y_train1 = traindata  # 真实标签
y_pred = testdata  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1值

# 打印NB模型的结果
print("NBresults")
print("accuracy")  # 打印准确率
print("%.3f" % accuracy)  # 保留三位小数的准确率
print("precision")  # 打印精确率
print("%.3f" % precision)  # 保留三位小数的精确率
print("racall")  # 打印召回率（拼写错误，应该是recall）
print("%.3f" % recall)  # 保留三位小数的召回率
print("f1score")  # 打印F1值
print("%.3f" % f1)  # 保留三位小数的F1值

# 读取RF模型的预测标签并重复上述过程
testdata = pd.read_csv('predictedlabelRF.txt', header=None)  # 读取RF模型的预测标签
y_train1 = traindata  # 真实标签
y_pred = testdata  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1值

# 打印RF模型的结果
print("RFresults")
print("accuracy")  # 打印准确率
print("%.3f" % accuracy)  # 保留三位小数的准确率
print("precision")  # 打印精确率
print("%.3f" % precision)  # 保留三位小数的精确率
print("racall")  # 打印召回率（拼写错误，应该是recall）
print("%.3f" % recall)  # 保留三位小数的召回率
print("f1score")  # 打印F1值
print("%.3f" % f1)  # 保留三位小数的F1值

# 读取SVM-rbf模型的预测标签并重复上述过程
testdata = pd.read_csv('predictedlabelSVM-rbf.txt', header=None)  # 读取SVM-rbf模型的预测标签
y_train1 = traindata  # 真实标签
y_pred = testdata  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1值

# 打印SVM-rbf模型的结果
print("SVM-rbfresults")
print("accuracy")  # 打印准确率
print("%.3f" % accuracy)  # 保留三位小数的准确率
print("precision")  # 打印精确率
print("%.3f" % precision)  # 保留三位小数的精确率
print("racall")  # 打印召回率（拼写错误，应该是recall）
print("%.3f" % recall)  # 保留三位小数的召回率
print("f1score")  # 打印F1值
print("%.3f" % f1)  # 保留三位小数的F1值
