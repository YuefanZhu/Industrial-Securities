# -*- coding: utf-8 -*-
# 连接wind
from WindPy import *

w.start()
w.isconnected()
from functools import reduce
import pandas as pd
import numpy as np
import math
import os

path = u'C:\\Users\\yz283\\Desktop\\ruitian'
write_file = u'C:\\Users\\yz283\\Desktop\\ruitian.xlsx'

# 取出文件夹下所有文件的名称
filename = pd.DataFrame()
filename['filename'] = os.listdir(path)
l = len(filename)
IndustrySummary = pd.DataFrame()
PESummary = pd.DataFrame()
MVSummary = pd.DataFrame()
Summary = pd.DataFrame(index=[u'股票个数', u'仓位(%)', "PE(TTM)", "PB", u'市值(亿)', u'个股最大持仓(%)', u'个股持仓中位数(%)'])

# 基金数据处理
for U in range(0, l):
    date = filename.loc[U, 'filename'][-13:-5]
    data = pd.read_excel(path + "\\" + filename.loc[U, 'filename'], sheetname='Sheet1', header=2)
    filename.loc[U, 'Date'] = date
    shstock = data.iloc[data[data[u'科目名称'] == u'上交所A股成本'].index[0] + 1:data[data[u'科目名称'] == u'上交所A股估值增值'].index[0] - 1,
          :]  # 取两个字符间的所有行，SH
    shstock['wind_code'] = shstock[u'科目代码'].apply(lambda x: x[-6:] + ".SH")
    szstock = data.iloc[data[data[u'科目名称'] == u'深交所A股成本'].index[0] + 1:data[data[u'科目名称'] == u'深交所A股估值增值'].index[0] - 1,
          :]  # 取两个字符间的所有行，SZ
    szstock['wind_code'] = szstock[u'科目代码'].apply(lambda x: x[-6:] + ".SZ")
    cystock = data.iloc[
          data[data[u'科目名称'] == u'深交所A股成本_创业板'].index[0] + 1:data[data[u'科目名称'] == u'深交所A股估值增值_创业板'].index[0] - 1,
          :]  # 取两个字符间的所有行，CYB
    cystock['wind_code'] = cystock[u'科目代码'].apply(lambda x: x[-6:] + ".SZ")

    stock = shstock.append(szstock)
    stock = stock.append(cystock)  # 合并数据
    stock = stock.reset_index()
    stock['Date'] = date
    # stock.to_excel(path+date+'_raw'+'.xlsx')

    # 360借壳把股票代码改了
    stock['wind_code'] = stock['wind_code'].apply(lambda x: '601360.SH' if x == '601313.SH' else x)
    wind_code = reduce(lambda x, y: x + ',' + y, stock['wind_code'])
    temp = w.wss(wind_code, "pe_ttm,pb,industry_sw,mkt_cap_CSRC", "tradeDate=" + date + ";ruleType=3;industryType=1;unit=1")
    winddata = pd.DataFrame(temp.Data, temp.Fields).T
    winddata['wind_code'] = temp.Codes
    stock = stock.merge(winddata, on='wind_code', how='outer')

    # 生成分类变量
    stock['PE_TYPE'] = stock['PE_TTM'].apply(lambda x: '<0' if x < 0 else '0-15' if x < 15 else '15-25' if x < 25 else \
        '25-40' if x < 40 else '40-50' if x < 50 else '50-100' if x < 100 else '>100')
    stock['MKT_CAP_CSRC'] = stock['MKT_CAP_CSRC'] / 100000000
    stock['MKT_CAP_TYPE'] = stock['MKT_CAP_CSRC'].apply(lambda
                                                        x: '<100' if x < 100 else '100-150' if x < 150  else '150-200' if x < 200 else '200-250' if x < 250  else '250-400' if x < 400  else '400-700' if x < 700 else'>700')

    # 计算中位数
    IndustrySummary[date] = stock.groupby('INDUSTRY_SW')[u'市值占净值(%)'].sum() / stock[u'市值占净值(%)'].sum() * 100
    PESummary[date] = stock.groupby('PE_TYPE')[u'市值占净值(%)'].sum() / stock[u'市值占净值(%)'].sum() * 100
    MVSummary[date] = stock.groupby('MKT_CAP_TYPE')[u'市值占净值(%)'].sum() / stock[u'市值占净值(%)'].sum() * 100

    Summary.loc[u'股票个数', date] = len(stock)
    Summary.loc[u'仓位(%)', date] = stock[u'市值占净值(%)'].sum()
    Summary.loc["PE(TTM)", date] = stock["PE_TTM"].median()
    Summary.loc["PB", date] = stock["PB"].median()
    Summary.loc[u'市值(亿)', date] = stock["MKT_CAP_CSRC"].median()
    Summary.loc[u'个股最大持仓(%)', date] = stock[u'成本占净值(%)'].max()
    Summary.loc[u'个股持仓中位数(%)', date] = stock[u'成本占净值(%)'].median()
    # print U

# 中证500数据处理
maxdate = filename['Date'].max()

zz500set = w.wset("sectorconstituent", "date=" + maxdate + ";windcode=399905.SZ")
zz500code = pd.DataFrame(zz500set.Data, index=zz500set.Fields).T

# Code格式转换，便于用wss函数提取多个代码的数据
wind_code = reduce(lambda x, y: x + ',' + y, zz500code['wind_code'])
temp = w.wss(wind_code, "pe_ttm,pb,industry_sw,mkt_cap_CSRC", "tradeDate=" + date + ";ruleType=3;industryType=1;unit=1")
zz500data = pd.DataFrame(temp.Data, temp.Fields).T
zz500data['wind_code'] = temp.Codes

# 生成分类变量
zz500data['PE_TYPE'] = zz500data['PE_TTM'].apply(lambda
                                                     x: '<0' if x < 0 else '0-15' if x < 15 else '15-25' if x < 25 else '25-40' if x < 40 else '40-50' if x < 50 else '50-100' if x < 100 else '>100')
zz500data['MKT_CAP_CSRC'] = zz500data['MKT_CAP_CSRC'] / 100000000
zz500data['MKT_CAP_TYPE'] = zz500data['MKT_CAP_CSRC'].apply(lambda
                                                                x: '<100' if x < 100 else '100-150' if x < 150  else '150-200' if x < 200 else '200-250' if x < 250  else '250-400' if x < 400  else '400-700' if x < 700 else'>700')
# 计算中位数
IndustrySummary[u'中证500'] = zz500data.groupby('INDUSTRY_SW')['MKT_CAP_CSRC'].sum() / zz500data[
    'MKT_CAP_CSRC'].sum() * 100
PESummary[u'中证500'] = zz500data.groupby('PE_TYPE')['MKT_CAP_CSRC'].sum() / zz500data['MKT_CAP_CSRC'].sum() * 100
MVSummary[u'中证500'] = zz500data.groupby('MKT_CAP_TYPE')['MKT_CAP_CSRC'].sum() / zz500data['MKT_CAP_CSRC'].sum() * 100

writer = pd.ExcelWriter(write_file)
Summary.to_excel(writer, 'Summary')
PESummary.to_excel(writer, 'PESummary')
MVSummary.to_excel(writer, 'MVSummary')
IndustrySummary.to_excel(writer, 'IndustrySummary')
writer.save()