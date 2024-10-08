import numpy as np  # 导入NumPy库，用于数值计算
import pandas as pd  # 导入Pandas库，用于数据处理
from sklearn.kernel_approximation import RBFSampler  # 导入RBF核近似采样器，用于核方法
from sklearn.linear_model import SGDClassifier  # 导入SGD分类器，基于随机梯度下降的分类器
from sklearn.model_selection import train_test_split  # 导入train_test_split函数，用于划分训练集和测试集
from sklearn import svm  # 导入SVM支持向量机模块
from sklearn.metrics import classification_report  # 导入classification_report，用于生成分类评估报告
from sklearn import metrics  # 导入metrics模块，用于评估模型
from sklearn.linear_model import LogisticRegression  # 导入Logistic回归模型
from sklearn.naive_bayes import GaussianNB  # 导入高斯朴素贝叶斯模型
from sklearn.neighbors import KNeighborsClassifier  # 导入K近邻分类器
from sklearn.tree import DecisionTreeClassifier  # 导入决策树分类器
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error)  # 导入评估指标
from sklearn.ensemble import AdaBoostClassifier  # 导入Adaboost分类器
from sklearn.ensemble import RandomForestClassifier  # 导入随机森林分类器
from sklearn.preprocessing import Normalizer  # 导入数据标准化工具
from sklearn.model_selection import GridSearchCV  # 导入GridSearchCV，用于网格搜索参数优化
from sklearn.svm import SVC  # 导入SVC，支持向量机分类器
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵，用于评估分类模型
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error, roc_curve, classification_report, auc)  # 导入评估指标和ROC曲线工具

# 读取训练数据和测试数据，header=None表示数据集没有标题行
traindata = pd.read_csv('train.csv', header=None)
testdata = pd.read_csv('test.csv', header=None)

# 提取特征和标签
X = traindata.iloc[:, 1:42]  # 提取训练数据的特征列（第1到41列）
Y = traindata.iloc[:, 0]  # 提取训练数据的标签列（第0列）
C = testdata.iloc[:, 0]  # 提取测试数据的标签列（第0列）
T = testdata.iloc[:, 1:42]  # 提取测试数据的特征列（第1到41列）

# 数据标准化处理，将特征值进行归一化
scaler = Normalizer().fit(X)  # 对训练数据进行归一化
trainX = scaler.transform(X)  # 归一化后的训练特征数据

scaler = Normalizer().fit(T)  # 对测试数据进行归一化
testT = scaler.transform(T)  # 归一化后的测试特征数据

# 将数据转化为NumPy数组
traindata = np.array(trainX)  # 训练特征数据
trainlabel = np.array(Y)  # 训练标签
testdata = np.array(testT)  # 测试特征数据
testlabel = np.array(C)  # 测试标签

# 决策树模型训练与评估
print("-----------------------------------------决策树DecisionTree---------------------------------")
model = DecisionTreeClassifier()  # 初始化决策树模型
model.fit(traindata, trainlabel)  # 使用训练数据拟合模型
print(model)

# 使用模型预测测试数据并计算预测概率
expected = testlabel  # 期望的真实标签
predicted = model.predict(testdata)  # 预测标签
proba = model.predict_proba(testdata)  # 预测的类别概率

# 保存预测标签和概率到文件
np.savetxt('classical/predictedlabelDT.txt', predicted, fmt='%01d')  # 保存决策树的预测标签
np.savetxt('classical/predictedprobaDT.txt', proba)  # 保存决策树的预测概率

# 计算模型评估指标
y_train1 = expected  # 真实标签
y_pred = predicted  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1分数

# 输出评估结果
print("----------------------------------------------")
print("accuracy")
print("%.3f" % accuracy)  # 输出准确率，保留3位小数
print("precision")
print("%.3f" % precision)  # 输出精确率，保留3位小数
print("racall")
print("%.3f" % recall)  # 输出召回率，保留3位小数（注：此处应为recall拼写错误）
print("f1score")
print("%.3f" % f1)  # 输出F1分数，保留3位小数

# 朴素贝叶斯模型训练与评估
print("-----------------------------------------朴素贝叶斯NaiveBayes---------------------------------")
model = GaussianNB()  # 初始化高斯朴素贝叶斯模型
model.fit(traindata, trainlabel)  # 使用训练数据拟合模型
print(model)

# 使用模型预测测试数据并计算预测概率
expected = testlabel  # 真实标签
predicted = model.predict(testdata)  # 预测标签
proba = model.predict_proba(testdata)  # 预测的类别概率

# 保存预测标签和概率到文件
np.savetxt('classical/predictedlabelNB.txt', predicted, fmt='%01d')  # 保存朴素贝叶斯的预测标签
np.savetxt('classical/predictedprobaNB.txt', proba)  # 保存朴素贝叶斯的预测概率

# 计算评估指标
y_train1 = expected  # 真实标签
y_pred = predicted  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1分数

# 输出评估结果
print("accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("racall")
print("%.3f" % recall)
print("f1score")
print("%.3f" % f1)

# K近邻模型训练与评估
print("-----------------------------------------K近邻KNN---------------------------------")
model = KNeighborsClassifier()  # 初始化K近邻模型
model.fit(traindata, trainlabel)  # 使用训练数据拟合模型
print(model)

# 使用模型预测测试数据并计算预测概率
expected = testlabel  # 真实标签
predicted = model.predict(testdata)  # 预测标签
proba = model.predict_proba(testdata)  # 预测的类别概率

# 保存预测标签和概率到文件
np.savetxt('classical/predictedlabelKNN.txt', predicted, fmt='%01d')  # 保存KNN的预测标签
np.savetxt('classical/predictedprobaKNN.txt', proba)  # 保存KNN的预测概率

# 计算评估指标
y_train1 = expected  # 真实标签
y_pred = predicted  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1分数

# 输出评估结果
print("----------------------------------------------")
print("accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("racall")
print("%.3f" % recall)
print("f1score")
print("%.3f" % f1)

# Adaboost模型训练与评估
print("-----------------------------------------Adaboost---------------------------------")
model = AdaBoostClassifier(n_estimators=100)  # 初始化Adaboost模型，使用100个弱分类器
model.fit(traindata, trainlabel)  # 使用训练数据拟合模型

# 使用模型预测测试数据并计算预测概率
expected = testlabel  # 真实标签
predicted = model.predict(testdata)  # 预测标签
proba = model.predict_proba(testdata)  # 预测的类别概率

# 保存预测标签和概率到文件
np.savetxt('classical/predictedlabelAB.txt', predicted, fmt='%01d')  # 保存Adaboost的预测标签
np.savetxt('classical/predictedprobaAB.txt', proba)  # 保存Adaboost的预测概率

# 计算评估指标
y_train1 = expected  # 真实标签
y_pred = predicted  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1分数

# 输出评估结果
print("----------------------------------------------")
print("accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("racall")
print("%.3f" % recall)
print("f1score")
print("%.3f" % f1)

# 随机森林模型训练与评估
print("--------------------------------------随机森林RandomForest--------------------------------------")
model = RandomForestClassifier(n_estimators=100)  # 初始化随机森林模型，使用100个决策树
model = model.fit(traindata, trainlabel)  # 使用训练数据拟合模型

# 使用模型预测测试数据并计算预测概率
expected = testlabel  # 真实标签
predicted = model.predict(testdata)  # 预测标签
proba = model.predict_proba(testdata)  # 预测的类别概率
np.savetxt('classical/predictedlabelRF.txt', predicted, fmt='%01d')  # 保存随机森林的预测标签
np.savetxt('classical/predictedprobaRF.txt', proba)  # 保存随机森林的预测概率

# 计算评估指标
y_train1 = expected  # 真实标签
y_pred = predicted  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1分数

# 输出评估结果
print("----------------------------------------------")
print("accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("racall")
print("%.3f" % recall)
print("f1score")
print("%.3f" % f1)

# 支持向量机（SVM）模型训练与评估
print("--------------------------------------SVM--------------------------------------")
model = svm.SVC(kernel='rbf', probability=True)  # 初始化支持向量机模型，使用RBF核，并启用概率预测
model = model.fit(traindata, trainlabel)  # 使用训练数据拟合模型

# 使用模型预测测试数据并计算预测概率
expected = testlabel  # 真实标签
predicted = model.predict(testdata)  # 预测标签
proba = model.predict_proba(testdata)  # 预测的类别概率
np.savetxt('classical/predictedlabelSVM-rbf.txt', predicted, fmt='%01d')  # 保存SVM的预测标签
np.savetxt('classical/predictedprobaSVM-rbf.txt', proba)  # 保存SVM的预测概率

# 计算评估指标
y_train1 = expected  # 真实标签
y_pred = predicted  # 预测标签
accuracy = accuracy_score(y_train1, y_pred)  # 计算准确率
recall = recall_score(y_train1, y_pred, average="binary")  # 计算召回率
precision = precision_score(y_train1, y_pred, average="binary")  # 计算精确率
f1 = f1_score(y_train1, y_pred, average="binary")  # 计算F1分数

# 输出评估结果
print("accuracy")
print("%.3f" % accuracy)
print("precision")
print("%.3f" % precision)
print("racall")
print("%.3f" % recall)
print("f1score")
print("%.3f" % f1)
