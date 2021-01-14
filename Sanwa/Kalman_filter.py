"""
Author： Ruming Liu & Zach

Kalman filter principle
# Reference can use Hamilton <Time Series Analysis Application>

y_t = A'*x_t + H'*z_t + \omega_t
z_t = F*z_{t-1} + v_t
where
y_t is the observable variable, the first equation is called the "space" or "observation" equation
x_t is exogenous (or predetermined) variable (we can set x_t = 0 for now)
z_t is the unobservable variable, the second equation is called "state" equation
v_t and  \omega_t are iid
E(v_t^2) = Q
E(\omega_t^2) = R

In finance, we want to estimate the parameters A, H, F, Q, R in order to understand where  the systmen is going if
given the states z_t.
In fact, we usually assume that x_t=1 and A=E(Y_t), or even that x_t=0

Note: any time series can be written as a state space
There is an example for AR(2)
state equation:
[Y_t+1 - \mu; Y_t - \mu] = [\Phi_1, \Phi_2; 1, 0][Y_{t} - \mu; Y_{t-1} - \mu] + [\epsilon_{t+1}; 0]
observation equation:
y_t = \mu + [1, 0][Y_{t+1}-\mu; Y_t-\mu]

As a first step, we will assume that A, H, F, Q, R are known
Our goal would be to find a best linear forecast of the state (unobserved) vector z_t. Such a forecast is needed in
control problems (to take decisions) and in finance (state of the economy, forecasts of unobserved volatility).

The forecasts will be denoted by:
z_{t+1|t} = E(z_{t+1}|y_t,...,x_t,...)
and we assume that we are only taking linear projections of z_{t+1} on y_t,...,x_t,...
Nonlinear Kalman Filters exist but the results are a bit more complicated

The Kalman Filter calculates the forecasts z_{t+1|t} recursively, starting with z1|0; then z2|1; ...until z_{T|T-1}

Since z_{t|t-1} is a forecast, we can ask how good of a forecast it is?
Therefore, we define P_{t|t-1} = E[(z_t - z_{t|t-1})(z_t - z_{t|t-1})], which is the forecasting error from the
recursive forecast z_{t|t-1}

The Kalman Filter can be broken down into 5 steps
1. Initialization of the recursion. Estimate z_{1|0}=E(z_1)
2. Forecasting y_t (intermediate step)
3. Updating Step
4. Forecast z_{t+1}|t
5. Go to step 2, until we reach T

Reference: http://stat.wharton.upenn.edu/~steele/Courses/434/434Context/PairsTrading/PairsTradingQFin05.pdf

"""

from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# This module uses Kalman Filter to estimate alpha & beta dynamicly between two stocks

def Kalman_a_b(stocki, stockj):
    cleanclose = pd.read_csv('clean_close_data'+'.csv', index_col=0)
    cleanclose.dropna(axis=1, inplace=True)
    stocklist = cleanclose.columns
    timeperiod = cleanclose.index
    time_length = len(timeperiod)
    stock_i = cleanclose[stocki]
    stock_j = cleanclose[stockj]
    time_1 = int(time_length*0.1) # use this time period for initial state estimation

    # create observation matrix
    observation_matrix = np.vstack(((np.ones(time_1)), stock_j.iloc[:time_1])).T
    Shape = observation_matrix.shape
    observation_matrix = observation_matrix.reshape(Shape[0], 1, Shape[1])
    # print(observation_matrix)

    # 定义卡尔曼滤波的方程
    kf = KalmanFilter(transition_matrices=np.array([[1, 0], [0, 1]]),  # 转移矩阵为单位阵
                      observation_matrices=observation_matrix)
    np.random.seed(0)

    # 使用2013年以前的数据，采用EM算法，估计出初始状态，
    # 初始状态的协方差，观测方程和状态方程误差的协方差
    kf.em(stock_i.iloc[:time_1])

    # 对time_1的数据做滤波
    filter_mean, filter_cov = kf.filter(stock_i.iloc[:time_1])

    # 从time_2开始滚动
    time_2 = time_1 + 1
    for i in range(time_2, time_length):
        observation_matrix = np.array([[1, stock_j.values[i]]])
        observation = stock_i.values[i]

        # 以上一个时刻的状态，状态的协方差以及当前的观测值，得到当前状态的估计
        next_filter_mean, next_filter_cov = kf.filter_update(
            filtered_state_mean=filter_mean[-1],
            filtered_state_covariance=filter_cov[-1],
            observation=observation,
            observation_matrix=observation_matrix)

        filter_mean = np.vstack((filter_mean, next_filter_mean))
        filter_cov = np.vstack((filter_cov, next_filter_cov.reshape(1, 2, 2)))

    # 得到alpha和beta
    alpha = pd.Series(filter_mean[time_2:, 0], index=cleanclose.index[time_2+1:])
    beta = pd.Series(filter_mean[time_2:, 1], index=cleanclose.index[time_2+1:])
    alpha_cov = pd.Series(filter_cov[time_2:, 0, 0], index=cleanclose.index[time_2+1:])
    alpha_minus_1sigma = abs(alpha - 1*alpha_cov**(0.5))
    alpha_plus_1sigma = abs(alpha + 1*alpha_cov**(0.5))
    print(filter_mean[time_2:])
    print("below is covariance")
    print(filter_cov[time_2:])

    real_spread = stock_i[time_2+1:] - beta*stock_j[time_2+1:]

    plt.plot(alpha[-30:],c='r')
    plt.plot(beta[-30:],c='g')
    plt.plot(real_spread[-30:])
    plt.title(stocki + " - " + stockj)
    plt.show()

    decision_df = pd.DataFrame([beta, alpha_minus_1sigma, alpha, alpha_plus_1sigma, real_spread],
                               index=["beta", "alpha-1sigma", "alpha", "alpha+1sigma", "real spread"]).T
    decision_df["spread too BIG"] = alpha_plus_1sigma < abs(decision_df['real spread'])
    decision_df["spread too SMALL"] = alpha_minus_1sigma > abs(decision_df['real spread'])
    decision_df["profit"] = 0

    for i in range(len(decision_df.index)-1):
        if decision_df["spread too BIG"].iloc[i] == True:
            decision_df.iloc[i, 4] = decision_df.iloc[i, 2] - decision_df.iloc[i+1, 2]
        if decision_df["spread too SMALL"].iloc[i] == True:
            decision_df.iloc[i, 4] = decision_df.iloc[i+1, 2] - decision_df.iloc[i, 2]

    plt.plot(decision_df["profit"])
    plt.title(stocki + " - " + stockj)
    plt.show()

    print(stocki + " - " + stockj + " profit is " + str(sum(decision_df["profit"])))
    decision_df.to_csv(stocki+"-"+stockj+".csv")

Kalman_a_b("GOOG", "GOOGL")

# # 对于可能的10组pairs做测试
# pair = 0
# cointegration_pair = pd.read_csv("cointegration_pairs.csv", index_col=0)
# for i in range(100):
#     possible_pair = cointegration_pair.index[i].split(" - ")
#     pair_i = possible_pair[0]
#     pair_j = possible_pair[1]
#     Kalman_a_b(pair_i, pair_j)
#     pair += 1
#     if pair == 20:
#         break
























