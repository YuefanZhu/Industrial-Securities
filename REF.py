# -*- coding: UTF-8 -*-
# 本策略为重抽样最优边界Resampled Efficient Frontier按收益率mu取最小方差再求均值
# 构造REF方法：分别求min Variance与max Return，两者的Variance作为两个端点，在其中等间隔取点求收益率的平均值

import pandas as pd
import numpy as np
from scipy import optimize
from gurobipy import *
from WindPy import *
w.start()

start_date = '2008-01-01'
end_date = '2018-07-27'
rtn = 0.06

# 上证综指、恒生指数、标普500、中证国债指数、中证信用债指数、南华商品指数、伦敦金、货币基金指数
windcode = ['000001.SH', 'HSI.HI', 'SPX.GI', 'H11006.CSI', 'H11073.CSI', 'NH0100.NHF', 'SPTAUUSDOZ.IDC', 'H11025.CSI']
dd = w.wsd(windcode, "close", start_date, end_date, "PriceAdj=F")
df = pd.DataFrame(dd.Data, index=windcode).T
df.index = [dd.Times]
df.dropna(inplace=True)
df.columns = ['CN', 'HK', 'US', 'Rate', 'Credit', 'Commodity', 'Gold', 'Monetary']
weight_m = pd.DataFrame(index=df.index, columns=df.columns)
# M对应月度换仓
df['M'] = [i.month for i in df.index.tolist()]

def minimize_Q(mu, Q, rtn):
    N = len(mu)
    model = Model()
    # Add variables to model
    y = []
    for j in xrange(N):
        y.append(model.addVar(vtype=GRB.CONTINUOUS))
        model.update()
    # Constraints
    expr = LinExpr()
    expr1 = LinExpr()
    for i in xrange(N):
        expr += mu[i] * y[i]
        expr1 += y[i]
    model.addConstr(expr, '==', rtn)
    model.addConstr(expr1, '==', 1)
    # Use full covariance matrix
    obj = QuadExpr()
    for i in xrange(N):
        for j in xrange(N):
            if Q[i][j] != 0:
                obj += Q[i][j] * y[i] * y[j]
    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    # Solve
    model.optimize()
    if model.status == GRB.Status.OPTIMAL:  # optimization status code
        result = model.getVars()
        yOpt = np.zeros(N)
        for i in xrange(N):
            yOpt[i] = result[i].x
        return yOpt / sum(yOpt)  # normalize to get weights
    else:
        return 'Not lucky this time'


def target_rtn(mu, sigma, rtn, n):
    sample = np.random.multivariate_normal(mu, sigma, n).tolist()
    weight = np.array([minimize_Q(muu, sigma, rtn).tolist() for muu in sample])
    return weight.mean(axis=0).tolist()


for i in range(122, df.shape[0]):
    # 月度换仓
    if df.M.iat[i] != df.M.iat[i - 1]:
        # 用半年度的收益率和协方差矩阵
        jj = 0
        for j in range(1, 200):
            if df.M.iat[i - j] != df.M.iat[i - j + 1]:
                jj += 1
            if jj == 7:
                break
        temp = df.iloc[(i - j + 1): i, :-1]
        # Resampled Efficient Frontier
        mu = (temp.iloc[-1, :] / temp.iloc[0, :] - 1).tolist()
        sigma = ((temp / temp.shift(1)).cov()*120).values.tolist()
        # 注意统一至半年度
        weight = target_rtn(mu, sigma, rtn / 2, 10)
        # T+1个交易日收盘后重置权重
        weight_m.iloc[i + 1, :] = weight

weight_m = weight_m.fillna(method = 'pad').dropna()

net_m = (((df/df.shift(1)).loc[weight_m.index, weight_m.columns]*weight_m).sum(axis = 1)).cumprod()
try:
    net_m.plot()
    weight_m.plot(kind = 'bar', stacked = True)
except:
    pass

df1 = df.loc[net_m.index, df.columns.tolist()[:(-1)]]
df1['net'] = net_m
# 策略评价
perf = pd.DataFrame()
perf['年化收益率'] = (df1.iloc[-1,:]/df1.iloc[0,:]-1)/(df1.index[-1]-df1.index[0]).days*365
perf['年化波动率'] = (df1/df1.shift(1)).std()*252**(1/2)
perf['夏普比率'] = perf['年化收益率']/perf['年化波动率']

max_point = df1.iloc[0,:]
max_dd = []
for ii in range(df1.shape[0]):
    max_point = pd.concat([max_point,df1.iloc[ii,:]],axis = 1).max(axis = 1)
    max_dd.append((df1.iloc[ii,:]/max_point-1).tolist())
max_dd = pd.DataFrame(max_dd,index = df1.index.tolist(), columns=df1.columns.tolist())
perf['基于日度数据的最大回撤'] = max_dd.min()

for ii in range(len(net_m)):
    if net_m.index[ii].year == net_m.index[-1].year:
        perf['年初至今收益'] = (df1.iloc[-1,:]/df1.iloc[(ii-1),:]-1)
        break
try:
    max_dd.net.plot()
except:
    pass

perf.index.name = df1.index[0].strftime('%Y-%m-%d')+' ~ '+end_date
print (perf.T)