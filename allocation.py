# -*- coding: UTF-8 -*-
# In python2.7
# 指数选取 沪深300、中证500、南华黄金、南华商品、中债国债、中债信用债
import pandas as pd
import numpy as np
import datetime
from gurobipy import *
from hmmlearn.hmm import GaussianHMM
from WindPy import *
w.start()

windcode = ['000300.SH','000905.SH','NH0008.NHF','NH0100.NHF','038.CS','066.CS']
Y = w.wsd(windcode, "close", '2008-01-11', datetime.now().strftime('%Y-%m-%d'))
Y = pd.DataFrame([Y.Times]+Y.Data, index = ['Date']+windcode).T
Y = pd.DataFrame(Y.iloc[:,1:].values,index = Y.iloc[:,0])
Y.columns = ['HS300','ZZ500','Gold','Commodity','Treasury','Credit']
df = (Y/Y.shift(1)-1).iloc[1:,:]

# initialize
roll = 120
perd = 60
vol = np.array(df.values, dtype = np.float64)
asset_weight = {}
netplot = pd.DataFrame(index = df.index[(roll+1):])

# 等权
equal = pd.DataFrame(np.average(vol, axis = 1), index = df.index)
netplot['Equal'] = equal.loc[netplot.index]

# 简单风险平价模型：假设每类资产相关性相同
# 收益率波动率倒数加权
weight = []
for i in xrange(roll, vol.shape[0], perd):
    temp = 1/np.std(vol[(i-roll):(i-1)], axis = 0)
    temp = temp / np.sum(temp)
    weight.append(temp.tolist())
weight = pd.DataFrame(weight, columns=df.columns, index = df.index[xrange(roll, vol.shape[0], perd)]).\
    loc[df.index].fillna(method='pad')
asset_weight['1/sigma'] = weight
net = pd.DataFrame(np.sum(weight.values * vol, axis = 1), index = df.index)
netplot['1/sigma'] = net.loc[netplot.index]
weight.plot()

# 收益率波动率EMA倒数加权
# k_0 * (r0-r_avg)_2 + k_1 * (r1-r_avg)_2 + k_2 * (r2-r_avg)_2...
weight=[]
for i in xrange(roll, vol.shape[0], perd):
    the = np.array([(roll/float(roll+1))**(roll-2-j) for j in xrange(roll - 1)])
    temp = vol[(i-roll):(i-1)]
    temp = (temp - np.average(temp, axis = 0))**2
    temp = np.sum(temp * the.reshape(roll-1, 1), axis = 0)
    temp = temp / np.sum(temp)
    weight.append(temp.tolist())
weight = pd.DataFrame(weight, columns=df.columns, index=df.index[xrange(roll, vol.shape[0], perd)]). \
    loc[df.index].fillna(method='pad')
asset_weight['1/sigma_ema'] = weight
net = pd.DataFrame(np.sum(weight.values * vol, axis = 1), index = df.index)
netplot['1/sigma_ema'] = net.loc[netplot.index]
weight.plot()

# ATR倒数加权
weight = []
for i in xrange(roll, vol.shape[0], perd):
    temp = vol[(i-roll):(i-1)]
    j = 0
    atr = np.max([np.max(temp[j:(j+5)], axis = 0) - np.min(temp[j:(j+5)], axis = 0),
                       np.abs(np.max(temp[j:(j + 5)], axis = 0) - temp[j]),
                       np.abs(np.min(temp[j:(j + 5)], axis = 0) - temp[j])],axis = 0)
    for j in xrange(5, roll, 5):
        atr = np.vstack([atr, np.max([np.max(temp[j:(j+5)], axis = 0) - np.min(temp[j:(j+5)], axis = 0),
                       np.abs(np.max(temp[j:(j + 5)], axis = 0) - temp[j - 1]),
                       np.abs(np.min(temp[j:(j + 5)], axis = 0) - temp[j - 1])],axis = 0)])
    temp = np.sum(atr, axis = 0)
    temp = temp / np.sum(temp)
    weight.append(temp.tolist())
weight = pd.DataFrame(weight, columns=df.columns, index=df.index[xrange(roll, vol.shape[0], perd)]). \
    loc[df.index].fillna(method='pad')
asset_weight['1/atr'] = weight
net = pd.DataFrame(np.sum(weight.values * vol, axis = 1), index = df.index)
netplot['1/atr'] = net.loc[netplot.index]
weight.plot()

# 优化风险平价模型：解最优化问题的数值解
# http://www.360doc.com/content/16/0728/20/35382359_579163965.shtml
def func(params, *args):
    # 提取维数
    n = args[0]
    # 提取协方差矩阵的参数
    for i in xrange(n*n):
        exec "q"+str(i)+" = args[i+1]"
    # 声明变量
    string=""
    for i in xrange(n):
        if i == n - 1:
            string += "A" + str(i) + "=params"
        else:
            string += "A"+str(i)+","
    exec string
    # 声明目标函数
    string="s_model = "
    for i in xrange(n):
        for j in xrange(n):
            string1 = "("
            string2 = "("
            for k in xrange(n):
                string1 += "+q" + str(i * n + k) + "*A" + str(k)
                string2 += "+q" + str(j * n + k) + "*A" + str(k)
            string1 += ")"
            string2 += ")"
            string += "+(A"+str(i)+"*"+string1+" - A"+str(j)+"*"+string2+" )**2"
    exec string
    return s_model

def risk_parity(Q):
    n = Q.shape[0]
    initial_values = np.ones(n)/2
    mybounds = [(0, 1)] * n
    x, f, d = scipy.optimize.fmin_l_bfgs_b(func, x0=initial_values, args=tuple([n]+(Q.reshape(Q.size, ) * 10 ** 5).tolist()),
                                           bounds=mybounds,
                                           approx_grad=True)
    return x/x.sum()

weight = []
for i in xrange(roll, vol.shape[0], perd):
    temp = risk_parity(np.cov(vol[(i-roll):(i-1)].T))
    weight.append(temp.tolist())
weight = pd.DataFrame(weight, columns=df.columns, index = df.index[xrange(roll, vol.shape[0], perd)]).\
    loc[df.index].fillna(method='pad')
asset_weight['RP'] = weight
net = pd.DataFrame(np.sum(weight.values * vol, axis = 1), index = df.index)
netplot['RP'] = net.loc[netplot.index]
weight.plot()

# mean-variance
weight = []
def sharpe_optimize(mu, Q):
    N = len(mu)
    model = Model()
    # Add variables to model
    y = []
    for j in xrange(N):
        y.append(model.addVar(vtype=GRB.CONTINUOUS))
        model.update()
    # Constraints
    expr = LinExpr()
    for i in xrange(N):
        expr += mu[i] * y[i]
    model.addConstr(expr, '==', 1)
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

for i in xrange(roll, vol.shape[0], perd):
    temp = vol[(i-roll):(i-1)]
    temp = sharpe_optimize(np.mean(temp, axis=0).tolist(), np.cov(temp.T).tolist())
    weight.append(temp.tolist())
weight = pd.DataFrame(weight, columns=df.columns, index=df.index[xrange(roll, vol.shape[0], perd)]). \
    loc[df.index].fillna(method='pad')
asset_weight['MV'] = weight
net = pd.DataFrame(np.sum(weight.values * vol, axis=1), index=df.index)
netplot['MV'] = net.loc[netplot.index]
weight.plot()

# 隐马尔可夫模型HMM
# notice: Markov Switching VAR != HMM
# https://zhuanlan.zhihu.com/p/20727973
# http://hmmlearn.readthedocs.io/en/latest/auto_examples/plot_hmm_stock_analysis.html
# Covariance of Mixture of Gaussian:
# https://math.stackexchange.com/questions/195911/covariance-of-gaussian-mixtures
roll1 = 1250
net = [1]
weight = []
for i in xrange(roll1, vol.shape[0] - perd, perd):
    vol1 = np.array(Y.values[(i - roll1 + 1):i:5], dtype=np.float64)
    # 用5天的对数收益率和方差来做更稳健
    # 输出时因为mu sigma^2都与t呈一次线性关系（mu,sigma^2 - t*mu,t*sigma^2），所以可直接用于1天的优化
    X = np.log(vol1[1:]) - np.log(vol1[:(len(vol1) - 1)])
    j = 2
    score = -1
    while True:
        hmm = GaussianHMM(n_components=j, covariance_type='full', n_iter=5000).fit(X)
        try:
            if hmm.score(X) <= score or j == 7:
                j = j - 1
                hmm = GaussianHMM(n_components=j, covariance_type='full', n_iter=5000).fit(X)
                break
            else:
                j = j + 1
                score = hmm.score(X)
        except:
            hmm = GaussianHMM(n_components=j, covariance_type='tied', n_iter=5000).fit(X)
            break
    print j
    # calculate the expected mean & variance
    # mu = p1 * mu1 + p2 * m2 + ......
    try:
        prob = hmm.transmat_[hmm.predict(X)[-1]]
    except:
        hmm = GaussianHMM(n_components=j, covariance_type='tied', n_iter=5000).fit(X)
        prob = hmm.transmat_[hmm.predict(X)[-1]]
    mu = (prob.reshape(len(prob), 1) * hmm.means_).sum(axis=0)
    Q = (prob.reshape(len(prob), 1, 1) * hmm.covars_).sum(axis=0) + \
        (prob.reshape(len(prob), 1, 1) * np.array(
            [(hmm.means_ - mu)[jj].reshape(hmm.means_.shape[1], 1) * (hmm.means_ - mu)[jj].reshape(1, hmm.means_.shape[1]) for jj in
             xrange(hmm.means_.shape[0])])).sum(axis=0)
    temp = sharpe_optimize(mu.tolist(), Q.tolist())
    weight.append(temp.tolist())
weight = pd.DataFrame(weight, columns=df.columns, index=df.index[xrange(roll1, vol.shape[0] - perd, perd)]). \
    loc[df.index].fillna(method='pad')
asset_weight['HMM'] = weight
net = pd.DataFrame(np.sum(weight.values * vol, axis=1), index=df.index)
netplot['HMM'] = net.loc[netplot.index]
weight.plot()

# 均线和波动率结合
net = [1]
weight=[]
st= []
ma1 = 5
ma2 = 20
stop_loss = 0.1
for i in xrange(roll, vol.shape[0], perd):
    temp = vol[(i-roll):i]
    temp = (np.average(temp[-ma1:], axis=0) / np.average(temp[-ma2:], axis=0) - 1) / np.std(temp / temp[0], axis=0)
    if temp[temp<0].shape == temp.shape:
        temp = np.ones_like(temp)
    temp[temp<0]=0
    temp = temp / np.sum(temp)
    weight.append([df.iat[i,0]]+temp.tolist())
    temp1 = vol[i:(i+perd)]/vol[i]
    for j in xrange(temp1.shape[1]):
        temp_max = 1
        for k in xrange(temp1.shape[0]-1):
            temp_max = max(temp_max, temp1[k+1,j])
            if temp1[k + 1, j] < (1-stop_loss)*temp_max:
                temp1[(k+1):,j] = temp1[k + 1, j]
                break
    net = net + (np.sum(temp*temp1, axis = 1)*net[-1]).tolist()
netplot['ma/sigma'] = net[1:]
weight = pd.DataFrame(weight,columns=df.columns)
weight.index = weight.Date
asset_weight['ma/sigma']=weight
weight.plot()

netplot.plot()
# 绩效评价
def perf(netplot):
    date = netplot.index.tolist()
    net = netplot.values
    perfm = pd.DataFrame()
    for i in xrange(net.shape[1]):
        # max drawdown
        m = 0
        dd = 0
        for j in xrange(len(net)):
            m = np.max([m, net[j, i]])
            dd = np.min([dd, (net[j, i] - m) / m])
        perfm[netplot.columns[i]] = \
        [(net[-1, i] / net[0, i]) ** (365.0 / (date[-1] - date[0]).days) - 1,
        np.std(net[1:, i] / net[:(len(net)-1), i]) * 252 ** (1 / 2.0),
        ((net[-1, i] / net[0, i]) ** (365.0 / (date[-1] - date[0]).days) - 1)/ \
         (np.std(net[1:, i] / net[:(len(net) - 1), i]) * 252 ** (1 / 2.0)),
        dd,
         -((net[-1, i] / net[0, i]) ** (365.0 / (date[-1] - date[0]).days) - 1)/dd]
    perfm.index = ['Annulized Return', 'Annulized Vol', 'Sharpe Ratio', 'Max DrawDown', 'Calmar Ratio']
    return perfm
perf(netplot)
perf(netplot).to_csv('C:\\Users\\yz283\\Desktop\\allco.csv')