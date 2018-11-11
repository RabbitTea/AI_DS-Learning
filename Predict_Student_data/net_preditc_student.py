import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *


"""
   首先要对于学生数据进行加载和预处理——
         1. 仅用gpa和gre绘制的散点图显示数据并不能因此有很好地分离；加入评级rank后
         观察到评级越低，录取率越高；所以加入rank作为输入的一个特征：一个输入总共获取到3个特征。
         2. 四个评级的数据分别是1,2,3,4，不方便神经网络对其处理，需要对其进行一次one-hot编码
         3. 观察到平时成绩(grades)和考试成绩(test scores)的范围相差很大，需要将这两个特征的数据放在0-1的范围内
         4. 将数据分为训练集和测试集(占总数据的10%)，测试集用以测试神经网络的泛化能力，防止过拟合。
         5. 将数据分为特征和标签(实际上是将输入和输出分离)，以便进行误差的计算和评估
"""

"""用pandas读取学生数据"""
data = pd.read_csv('student_data.csv')


# 仅根据gre和gpa变量绘制数据的散点图
def plot_points(data):
    X = np.array(data[["gre", "gpa"]])
    y = np.array(data["admit"])
    admitted = X[np.argwhere(y == 1)]  # 保存被录取的学生的测试成绩和平时成绩
    rejected = X[np.argwhere(y == 0)]  # 保存被拒绝的学生的测试成绩和平时成绩
    # 绘制散点图
    plt.scatter([s[0][0] for s in rejected], [s[0][1] for s in rejected], s=25, color='red', edgecolor='k')
    plt.scatter([s[0][0] for s in admitted], [s[0][1] for s in admitted], s=25, color='cyan', edgecolor='k')
    plt.xlabel('Test (GRE)')
    plt.ylabel('Grades (GPA)')

# 显示散点图
# plot_points(data)
# plt.show()


# 加入评级rank标准来绘制数据图
def plot_points_rank(data):
    data_rank1 = data[data["rank"] == 1]
    data_rank2 = data[data["rank"] == 2]
    data_rank3 = data[data["rank"] == 3]
    data_rank4 = data[data["rank"] == 4]

    # 图片显示
    plot_points(data_rank1)
    plt.title("Rank 1")
    plt.show()
    plot_points(data_rank2)
    plt.title("Rank 2")
    plt.show()
    plot_points(data_rank3)
    plt.title("Rank 3")
    plt.show()
    plot_points(data_rank4)
    plt.title("Rank 4")
    plt.show()


plot_points_rank(data)


"""绘制当前的分割直线"""
def display(m, b, color='g--'):
    plt.xlim(-0.05,1.05)
    plt.ylim(-0.05,1.05)
    x = np.arange(-10, 10, 0.1)
    plt.plot(x, m*x+b, color)


"""使用pandas中的get_dummies方法对评级数据进行one-hot编码"""
one_hot_data = pd.concat([data, pd.get_dummies(data['rank'], prefix='rank')], axis=1)
one_hot_data = one_hot_data.drop('rank', axis=1)
# print(one_hot_data[:10])


"""缩放数据——将平时成绩(grades)除以4.0，将测试成绩(test scores)除以800"""
processed_data = one_hot_data[:]
processed_data['gre'] = processed_data['gre']/800
processed_data['gpa'] = processed_data['gpa']/4.0
# print(processed_data[:10])


"""将数据分成训练集和测试集"""
sample = np.random.choice(processed_data.index, size=int(len(processed_data)*0.9), replace=False)
train_data, test_data = processed_data.iloc[sample], processed_data.drop(sample)

# print("Number of training samples is", len(train_data))
# print("Number of testing samples is", len(test_data))
# print(train_data[:10])
# print(test_data[:10])


"""将数据分成特征和标签(输入和输出的分离)"""
features = train_data.drop('admit', axis=1)
targets = train_data['admit']
features_test = test_data.drop('admit', axis=1)
targets_test = test_data['admit']

# print(features[:10])
# print(targets[:10])


"""计算误差项"""
def error_term_formula(y, output):
    return (y-output) * output * (1 - output)


"""梯度下降法进行更新"""
# 超参数
epochs = 1000  # 每隔一个新纪元更新一次
learnrate = 0.5  # 学习率设定为0.5

"""
# 训练函数
def train_network_bp(features, targets, epochs, learnrate):
    # 使用相同的整数值使debug更简单
    np.random.seed(42) # seed()用于指定随机数生成时所用算法开始的整数值

    n_records, n_features = features.shape
    last_loss = None

    # 初始化权重参数
    # bias = 0
    W = np.random.normal(scale=1 / n_features ** .5, size=n_features)
    # display(-W[0] / W[1], -bias / W[1])  # 画当前求解出来的分界线

    for e in range(epochs):
        del_w = np.zeros(W.shape)
        for x, y in zip(features.values, targets):
            # x是输入，y是标签

            output = sigmoid(np.dot(x, W))  # 获取神经网络的输出

            # 用交叉熵来计算损失函数
            error = error_formula(y, output)

            # 误差项
            # 反向误差传播的方法比数值微分法更快，因为我们重复利用了sigmoid函数的输出结果(保存在output中)
            error_term = error_term_formula(y, output)

            # 得到最终的偏导结果(误差项传播到了最开始的输入)
            del_w += error_term * x

        # 更新权重参数
        W += learnrate * del_w / n_records

        # 输出训练集的均方误差
        if e % (epochs / 10) == 0:
            out = sigmoid(np.dot(features, W))
            loss = np.mean((out - targets) ** 2)
            # print("Epoch:", e)
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            print("=========")
    print("Finished training!")
    return W


W = train_network_bp(features, targets, epochs, learnrate)
"""
#训练函数，用于训练分界线
def train(features, targets, epochs, learnrate, graph_lines=False):
    errors = []
    n_records, n_features = features.shape#n_records=100,n_features=2
    last_loss = None
    weights = np.random.normal(scale=1 / n_features ** .5, size=n_features) #初始值用随机数生成权重 2*1
    bias = 0
    display(-weights[0] / weights[1], -bias / weights[1])  # 画当前求解出来的分界线

    for e in range(epochs): #迭代1000次
        del_w = np.zeros(weights.shape)
        for x, y in zip(features, targets):#通过zip拉锁函数将X与y的每个点结合起来
            output = sigmoid(np.dot(x, weights)) #计算输出预测值yi 其中x是1*2，weight是2*1
            error = error_formula(y, output)#计算每一个yi的误差
            error_term = error_term_formula(y, output)
            del_w += error_term * x
            weights += learnrate * del_w / n_records

        print(weights,bias)
        print(e)#注意 每次迭代里都对xi即100组数进行计算都更新了权重，即更新了100*迭代次数次，每次迭代都是以上次的结果重新计算100组数
        # Printing out the log-loss error on the training set
        out = sigmoid(np.dot(x, weights))#计算迭代后的预测值，这里feature是n*2,weight是2*1，out是n*1的一列预测值
        loss = np.mean(error_formula(targets, out))#对每个预测值的误差做算术平均
        errors.append(loss)
        if e % (epochs / 10) == 0:
            print("\n========== Epoch", e, "==========")
            if last_loss and last_loss < loss:
                print("Train loss: ", loss, "  WARNING - Loss Increasing")
            else:
                print("Train loss: ", loss)
            last_loss = loss
            predictions = out > 0.5
            accuracy = np.mean(predictions == targets)
            print("Accuracy: ", accuracy)
        if graph_lines :#and e % (epochs / 100) == 0
            display(-weights[0] / weights[1], -bias / weights[1])#画当前求解出来的分界线

    # Plotting the solution boundary
    plt.title("Solution boundary")
    display(-weights[0] / weights[1], -bias / weights[1], 'black')#画最后一根求解出来的分界线

    # Plotting the data
    plot_points(features, targets)
    plt.show()

    # Plotting the error
    plt.title("Error Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Error')
    plt.plot(errors)
    plt.show()
    return weights

W = train(features, targets, epochs, learnrate)


"""计算测试数据的精确度"""
tes_out = sigmoid(np.dot(features_test, W))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
