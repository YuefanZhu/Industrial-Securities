# -*- coding: UTF-8 -*-
# 本策略为比较风险平价模型（Risk Parity）、分层风险平价模型（Hierarchical Risk Parity）应用在中国市场的差异
# Hierarchical Risk Parity 参考  Building diversified portfolios that outperform out of sample
# ———————————————————————————————————————————————————————————
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy import optimize
from WindPy import *
w.start()
# ———————————————————————————————————————————————————————————
# Hierarchical Risk Parity
def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp

def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar

def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()

def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w

def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.)**.5  # distance matrix
    return dist

def getHRP(cov, corr):
    # Construct a hierarchical portfolio
    dist = correlDist(corr)
    link = sch.linkage(dist, 'single')
    #dn = sch.dendrogram(link, labels=cov.index.values, label_rotation=90)
    #plt.show()
    sortIx = getQuasiDiag(link)
    sortIx = corr.index[sortIx].tolist()
    hrp = getRecBipart(cov, sortIx)
    return hrp.sort_index()
# ———————————————————————————————————————————————————————————
# Risk Parity
def func(params, *args):
    # 提取维数
    n = args[0]
    # 提取协方差矩阵的参数
    for i in range(n*n):
        exec "q"+str(i)+" = args[i+1]"
    # 声明变量
    string=""
    for i in range(n):
        if i == n - 1:
            string += "A" + str(i) + "=params"
        else:
            string += "A"+str(i)+","
    exec string
    # 声明目标函数
    string="s_model = "
    for i in range(n):
        for j in range(n):
            string1 = "("
            string2 = "("
            for k in range(n):
                string1 += "+q" + str(i * n + k) + "*A" + str(k)
                string2 += "+q" + str(j * n + k) + "*A" + str(k)
            string1 += ")"
            string2 += ")"
            string += "+(A"+str(i)+"*"+string1+" - A"+str(j)+"*"+string2+" )**2"
    exec string
    return s_model

def getRP(Q):
    Q = Q.values
    n = Q.shape[0]
    initial_values = np.ones(n)/2
    mybounds = [(0, 1)] * n
    x, f, d = optimize.fmin_l_bfgs_b(func, x0=initial_values, args=tuple([n]+(Q.reshape(Q.size, ) * 10 ** 10).tolist()),
                                           bounds=mybounds,
                                           approx_grad=True)
    return x/x.sum()
# ———————————————————————————————————————————————————————————
# import data from Wind Api
start_date = '2008-01-01'
end_date = '2018-09-28'
windcode = ['000001.SH', 'HSI.HI', 'SPX.GI', 'H11006.CSI', 'H11073.CSI', 'NH0100.NHF', 'SPTAUUSDOZ.IDC']
dd = w.wsd(windcode, "close", start_date, end_date, "PriceAdj=F")
df = pd.DataFrame(dd.Data, index=windcode).T
df.index = [dd.Times]
df.dropna(inplace=True)
df.columns = ['CN', 'HK', 'US', 'Rate', 'Credit', 'Commodity', 'Gold']
weight_RP, weight_HRP = [pd.DataFrame(index=df.index, columns=df.columns) for _ in range(2)]
df['M'] = [i.month for i in df.index.tolist()]
# ———————————————————————————————————————————————————————————
# Back test
for i in xrange(122, df.shape[0]):
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
        # RP
        weight = getRP(((temp / temp.shift(1)).cov() * 120))
        weight_RP.iloc[i + 1, :] = weight
        # HRP
        weight = getHRP(((temp / temp.shift(1)).cov()*120), (temp / temp.shift(1)).corr()).reindex(temp.columns)
        weight_HRP.iloc[i + 1, :] = weight
# Weightings for RP、HRP
weight_RP = weight_RP.fillna(method = 'pad').dropna()
weight_HRP = weight_HRP.fillna(method = 'pad').dropna()
# weight_RP.plot(kind = 'bar', stacked = True)
# weight_HRP.plot(kind = 'bar', stacked = True)

# Calculate net value
net = pd.DataFrame()
net['Risk Parity'] = (((df/df.shift(1)).loc[weight_RP.index, weight_RP.columns]*weight_RP).sum(axis = 1)).cumprod()
net['Hierarchical Risk Parity'] = (((df/df.shift(1)).loc[weight_HRP.index, weight_HRP.columns]*weight_HRP).
                                   sum(axis = 1)).cumprod()
net.plot()
# ———————————————————————————————————————————————————————————
# 策略评价部分
perf = pd.DataFrame()
perf['年化收益率'] = (net.iloc[-1,:]/net.iloc[0,:]-1)/(net.index[-1]-net.index[0]).days*365
# 注意power用小数
perf['年化波动率'] = (net/net.shift(1)).std()*pow(252,0.5)
perf['夏普比率'] = perf['年化收益率']/perf['年化波动率']
# 计算各策略的动态回撤
max_point = net.iloc[0,:]
max_dd = []
for ii in xrange(net.shape[0]):
    max_point = pd.concat([max_point,net.iloc[ii,:]],axis = 1).max(axis = 1)
    max_dd.append((net.iloc[ii,:]/max_point-1).tolist())
max_dd = pd.DataFrame(max_dd,index = net.index.tolist(), columns=net.columns.tolist())
perf['基于日度数据的最大回撤'] = max_dd.min()
# 各策略动态回撤示意图
# max_dd.plot()

# 计算年初至今收益
for ii in xrange(len(net)):
    if net.index[ii].year == net.index[-1].year:
        perf['年初至今收益'] = (net.iloc[-1,:]/net.iloc[(ii-1),:]-1)
        break
# 标注时间
perf.index.name = net.index[0].strftime('%Y-%m-%d')+' ~ '+end_date
print perf.T