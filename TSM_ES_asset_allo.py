# -*- coding: UTF-8 -*-
# 本策略复制安信证券-20171016《风险再平衡-基于时序动量的大类资产配置策略》
# ———————————————————————————————————————————————————————————
import pandas as pd
from scipy.stats import norm, t
from WindPy import *
w.start()
# ———————————————————————————————————————————————————————————
# import data from Wind Api
start_date = '2005-01-01'
end_date = '2018-09-28'
windcode = ['000001.SH', 'HSI.HI', 'SPX.GI', 'H11006.CSI', 'NH0100.NHF', 'SPTAUUSDOZ.IDC']
dd = w.wsd(windcode, "close", start_date, end_date, "PriceAdj=F")
df = pd.DataFrame(dd.Data, index=windcode).T
df.index = [dd.Times]
df.dropna(inplace = True)
df.rename(columns={'000001.SH':'CN_Equity','HSI.HI':'HK_Equity','SPX.GI':'US_Equity',
                   'H11006.CSI': 'Rate','NH0100.NHF':'Commodity','SPTAUUSDOZ.IDC':'Gold'},
          inplace = True)
# ———————————————————————————————————————————————————————————
# test SMA strategies on each asset
def SMA(Df, n = 6):
    Df = Df.to_frame()
    Df['M'] = [i.month for i in Df.index]
    mth = [1, 3, 6, 9, 10]
    weight = pd.DataFrame(index = Df.index, columns = map(lambda x: 'SMA'+str(x), mth))
    for i in xrange(240, Df.shape[0]):
        if Df.M[i] != Df.M[i - 1]:
            for j in xrange(len(mth)):
                temp = 0
                for k in xrange(1, 300):
                    if Df.M.iat[i - k] != Df.M.iat[i - k + 1]:
                        temp += 1
                    if temp == mth[j] + 1:
                        break
                temp = Df.iloc[(i - k + 1): i, :-1]
                weight.iat[i + 1, j] = 1. if (Df.iat[i - 1, 0] > temp.mean())[0] else 0.
    weight = weight.fillna(method = 'pad').dropna()
    weight[Df.columns[0]] = [1.] * weight.shape[0]
    del Df['M']
    net = weight.apply(lambda x : x * (Df/Df.shift(1)).loc[weight.index, :].iloc[:,0]).\
        applymap(lambda x: 1. if x == 0 else x).cumprod()
    # net.plot()
    return net.iloc[:, mth.index(n)]
# ———————————————————————————————————————————————————————————
# Expected Shortfall(CVar) Strategy
def calc_CVar(Df, alpha = 0.05, dist = 'n'):
    rtn = Df/Df.shift(1) - 1
    if dist == 'n':
        mu, sig = norm.fit(rtn.dropna().values)
        mu = mu * 252
        sig = sig * 252 ** (0.5)
        CVar = alpha ** (-1.) * norm.pdf(norm.ppf(alpha))*sig - mu
    if dist == 't':
        nu, mu, sig = t.fit(rtn.dropna().values)
        mu = mu * 252
        sig = sig * 252 ** (0.5)
        xanu = t.ppf(alpha, nu)
        CVar = -1. / alpha * (1 - nu) ** (-1.) * (nu - 2 + xanu ** 2.) * t.pdf(xanu, nu) * sig - mu
    return CVar

def ES(Df, target_ES = 0.06):
    weight = pd.DataFrame(index=Df.index, columns=Df.columns)
    Df['M'] = [i.month for i in Df.index]
    for i in xrange(240, Df.shape[0]):
        if Df.M[i] != Df.M[i - 1]:
            temp = Df.iloc[(i - 240): i, :-1]
            wht = [1./weight.shape[1] if jj < target_ES else 1./weight.shape[1] * target_ES/jj
                   for jj in temp.apply(calc_CVar, axis = 0).tolist()]
            # 债券ES设置为2%
            wht[3] = 1./weight.shape[1] if calc_CVar(temp.iloc[:, 3]) < 0.02 \
                else 1./weight.shape[1] * 0.02/calc_CVar(temp.iloc[:, 3])
            wht1 = [1. if jj == 1./weight.shape[1] else 0. for jj in wht]
            if sum(wht1) > 0:
                wht = [wht[jj]+wht1[jj]*(1-sum(wht))/sum(wht1) for jj in xrange(len(wht))]
            weight.iloc[i + 1, :] = wht
    weight = weight.fillna(method='pad').dropna()
    # 注意weight在某些时刻的和不为1
    net = ((((Df/Df.shift(1)-1).loc[weight.index, weight.columns] * weight).sum(axis = 1))+1).cumprod()
    # net.plot()
    del Df['M']
    return net, weight
# ———————————————————————————————————————————————————————————
# TSM Strategy: TSM因子确定权重
def TSM(Df):
    weight = pd.DataFrame(index = Df.index, columns = Df.columns)
    Df['M'] = [i.month for i in Df.index]
    for i in range(240, Df.shape[0]):
        if Df.M[i] != Df.M[i - 1]:
            temp = Df.iloc[(i - 240): i, :-1]
            wht = (temp.iloc[-1, :]/temp.iloc[-90, :]-1)
            wht[wht < 0.] = 0.
            wht = wht / sum(wht)
            weight.iloc[i + 1, :] = wht
    weight = weight.fillna(method='pad').dropna()
    # 注意weight在某些时刻的和不为1
    net = ((((Df/Df.shift(1)-1).loc[weight.index, weight.columns] * weight).sum(axis = 1))+1).cumprod()
    # net.plot()
    del Df['M']
    return net, weight
# ———————————————————————————————————————————————————————————
# TSM+ES Strategy: TSM因子确定初始权重，然后根据ES调整
def TSM_ES(Df, target_ES = 0.06):
    weight = pd.DataFrame(index = Df.index, columns = Df.columns)
    Df['M'] = [i.month for i in Df.index]
    for i in xrange(240, Df.shape[0]):
        if Df.M[i] != Df.M[i - 1]:
            temp = Df.iloc[(i - 240): i, :-1]
            wht = (temp.iloc[-1, :]/temp.iloc[-90, :]-1)
            wht[wht < 0.] = 0.
            wht = wht / sum(wht)
            ES = temp.apply(calc_CVar, axis = 0).tolist()
            wht1 = [wht[jj] if ES[jj] < target_ES else wht[jj] * target_ES/ES[jj]
                   for jj in xrange(temp.shape[1])]
            # 债券ES设置为2%
            wht1[3] = wht[3] if ES[3] < 0.02 else wht[3] * 0.02 / ES[3]
            wht2 = [1. if wht[jj] == wht1[jj] else 0. for jj in xrange(temp.shape[1])]
            if sum(wht2) > 0:
                wht1 = [wht1[jj] + wht2[jj] * (1-sum(wht1))/sum(wht2) for jj in xrange(len(wht1))]
            weight.iloc[i + 1, :] = wht1
    weight = weight.fillna(method='pad').dropna()
    # 注意weight在某些时刻的和不为1
    net = ((((Df/Df.shift(1)-1).loc[weight.index, weight.columns] * weight).sum(axis = 1))+1).cumprod()
    # net.plot()
    del Df['M']
    return net, weight
# ———————————————————————————————————————————————————————————
# Global Tactical Asset Allocation Strategy - GTAA
gtaa = df.apply(SMA, axis = 0)
gtaa = gtaa.mean(axis = 1)
# Expected Shortfall Strategy
es,wht_es = ES(df)
# wht_es.plot(kind = 'bar', stacked = True)
# TSM factor
tsm,wht_tsm = TSM(df)
# wht_tsm.plot(kind = 'bar', stacked = True)
# TSM factor + ES Strategy
tsmes,wht_tsmes = TSM_ES(df)
# wht_tsmes.plot(kind = 'bar', stacked = True)
# net value
net = pd.concat([gtaa, es, tsm, tsmes], axis = 1)
net.columns = ['GTAA','ES','TSM','TSM_ES']
net.plot()
# ———————————————————————————————————————————————————————————
# Performance Measurement
perf = pd.DataFrame()
perf['年化收益率'] = (net.iloc[-1,:]/net.iloc[0,:]) ** (365./(net.index[-1]-net.index[0]).days) - 1
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
