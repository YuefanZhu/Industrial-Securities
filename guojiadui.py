# -*- coding:utf8 -*-
import numpy as np
import pandas as pd
import xlrd
import glob
import datetime

market_data_dir = 'C:\\Users\\yz283\\Desktop\\market.xlsx'

excel = xlrd.open_workbook(market_data_dir)
sheet = excel.sheets()[0]
col_names = ['windcode','stock','price_171031','shares','pe','eps']
market = pd.DataFrame(sheet._cell_values[1:-3], columns= col_names).convert_objects(convert_numeric=True)
market['market_cap'] = market.price_171031 * market.shares

dir = 'C:\\Users\\yz283\\Desktop\\data'
# dir = dir.decode('utf-8')
suffix = 'xlsx'
f = glob.glob(dir + '\\*.' + suffix)

excel = xlrd.open_workbook(f[0])
sheet = excel.sheets()[0]
col_names = sheet.row_values(1)
data = np.array(sheet._cell_values[2:-2])

for file_index in f[1:]:
    excel = xlrd.open_workbook(file_index)
    sheet = excel.sheets()[0]
    data = np.vstack((data,np.array(sheet._cell_values[2:-2])))

data = pd.DataFrame(data, columns=col_names).iloc[:,[0,1,2,3,6,9,21,22,23]]
data.columns = ['windcode','name','institution','volume','holdings_of_market_value','proportion_of_shares_in_circulation',
                'wind_industry_classification','sec_industry_classification','Date']

# excel中1900-01-01设定为1
data.Date = [datetime.datetime.strptime('1900-01-01','%Y-%m-%d') + datetime.timedelta(days=int(ii.split('.')[0])-2) for ii in data.Date.tolist()]
data = data.join(market.set_index('windcode'), on='windcode').convert_objects(convert_numeric=True)
# data.to_csv('C:\\Users\\yz283\\Desktop\\data.csv', encoding='utf_8_sig')

# data = pd.read_csv('C:\\Users\\yz283\\Desktop\\data.csv', encoding='utf_8_sig').iloc[:,1:]
data['holdings_of_market_value_171031'] = data.volume * data.price_171031
data[['holdings_of_market_value','holdings_of_market_value_171031','volume','Date']].groupby(['Date']).sum()


# 进行跨季度比较需要剔除价格变化因素
data[['holdings_of_market_value_171031','Date','wind_industry_classification']].groupby(['Date','wind_industry_classification']).sum().unstack()

pe_bins = [-100000,0,5,10,15,20,25,30,35,40,45,50,75,100,150,200,100000]
eps_bins = [-100000.0,0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.5,100000.0]
data.pe = pd.cut(data.pe.tolist(),pe_bins)
data.eps = pd.cut(data.eps.tolist(),eps_bins)
market.pe = pd.cut(market.pe.tolist(),pe_bins)
market.eps = pd.cut(market.eps.tolist(),eps_bins)
data[['eps','pe','Date','holdings_of_market_value_171031']].groupby(['pe','Date']).sum()
market[['eps','pe','market_cap']].groupby(['pe']).sum()

data[['eps','pe','Date','holdings_of_market_value_171031']].groupby(['eps','Date']).sum()
market[['eps','pe','market_cap']].groupby(['eps']).sum()