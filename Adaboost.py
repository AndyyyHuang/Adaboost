"""
# File         :Adaboost.py
# Time         :2022/5/14 22:38
# Author       :Andy
# contact      :2019200836@ruc.edu.cn
# version      :python 3.9
# Description:
"""




from sklearn.metrics import confusion_matrix
import pandas as pd
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

def get_Datasets():
    from sklearn.datasets import make_classification
    dataSet, classLabels = make_classification(n_samples=200, n_features=100, n_classes=2)
    #print(dataSet.shape,classLabels.shape)
    return np.concatenate((dataSet, classLabels.reshape((-1, 1))), axis=1)

def get_score(y_test, y_pred):
    """

    :param y_test:
    :param y_pred:
    :return:
    """
    confusion_ = confusion_matrix(y_test, y_pred)
    confusion_ = pd.DataFrame(confusion_, index=[-1, 1], columns=[-1, 1])
    confusion_.index.name = 'actual'
    confusion_.columns.name = 'pred'
    Accuracy = (confusion_.iloc[0, 0] + confusion_.iloc[1, 1]) / confusion_.sum().sum()
    Precison = confusion_.iloc[1, 1] / confusion_.sum()[1]
    Recall = confusion_.iloc[1, 1] / confusion_.sum(axis=1)[1]
    F1_score = 2 * Precison * Recall / (Precison + Recall)
    print("Accuracy is {}\nPrecision is {}\nRecall is {}\nF1_score is {}"
          .format(Accuracy, Precison, Recall, F1_score))
    return Accuracy





def iter_adaboost(model, X_train, y_train, X_test, y_test):
    global step
    global sample_lis
    global sample_weight
    global pred_lis
    global score_lis
    global train_model_lis
    global final_accu_lis
    global alpha_lis
    print("第{}轮训练开始".format(step))
    model.fit(X_train, y_train, sample_weight=sample_weight)

    pred = model.predict(X_train)
    pred_lis.append(pred)
    score = model.score(X_train, y_train)
    print("new model score is {}".format(score))
    actual_lis = y_train
    # 判断对误的值储存
    accu_lis = [1 if pred[i] == actual_lis[i]
                else -1 for i in range(len(pred))]
    # 获取分类器权重
    alpha = np.log(score/(1-score))/2
    print("alpha is {}".format(alpha))
    # 调整样本权重
    sample_weight = [sample_weight[i]*np.exp(-alpha) if accu_lis[i] == 1
                     else sample_weight[i]*np.exp(alpha) for i in range(len(sample_weight))]
    sample_weight = (sample_weight/sum(sample_weight)) * X_train.shape[0]
    # 样本权重和模型权重的储存
    sample_lis.append(sample_weight)
    alpha_lis.append(alpha)
    # 强学习器结果聚合
    final_res = np.full(shape=X_train.shape[0], fill_value=0)
    for i in range(step):
        final_res = final_res + (np.array(pred_lis[i]) * alpha_lis[i])

    result = [1 if i > 0 else -1 for i in final_res]
    confusion_ = confusion_matrix(y_train, result)
    confusion_ = pd.DataFrame(confusion_, index=[0,1], columns=[0,1])
    final_score = (confusion_.iloc[0,0]+confusion_.iloc[1,1]) / confusion_.sum().sum()
    # 得分的存储
    score_lis.append(final_score)
    print("强学习器得分为{}".format(final_score))
    # 模型的储存
    train_model_lis.append(model)

    # predict in the test
    final_res = np.full(shape=X_test.shape[0], fill_value=0)
    for i in range(len(train_model_lis)):
        model = train_model_lis[i]
        pred_ = model.predict(X_test)
        final_res = final_res + np.array(pred_) * alpha_lis[i]
    result = np.array([1 if i > 0 else -1 for i in final_res])
    final_accu = get_score(y_test, result)
    final_accu_lis.append(final_accu)
    return alpha, final_score, sample_weight, train_model_lis



if __name__ == "__main__":
    data = get_Datasets()
    X_train, X_test, y_train, y_test = train_test_split(data[:, :-1], data[:, -1], test_size=0.2)
    step = 1
    max_iter = 50
    score_lis = []
    sample_weight = np.full(shape=X_train.shape[0], fill_value=1)
    sample_lis = []
    alpha_lis = []
    pred_lis = []
    train_model_lis = []
    final_accu_lis = []
    improve = 1
    alpha = 1
    while step <= max_iter and alpha > 0:
        # 记录下每一个iteration的aplha、model、sample_weight 方便最后选择最佳模型。
        alpha, final_score, sample_weight, train_model_lis = iter_adaboost(model=BernoulliNB(), X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
        step += 1