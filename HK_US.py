# encoding: utf-8
from WindPy import *
import pandas as pd
import datetime
w.start()
threshold = 0.1

date_now = datetime.datetime.now()
end_day = date_now - datetime.timedelta(days = ((date_now.weekday() <= 3) and date_now.weekday() + 3 or
                                                date_now.weekday() - 4))
start_day = end_day - datetime.timedelta(days = 4)
start_day, end_day = start_day.strftime('%Y-%m-%d'), end_day.strftime('%Y-%m-%d')

def get_price(windcode):
    temp = w.wsd(windcode[1], "open,close", start_day, end_day)
    return windcode + [temp.Data[1][-1], round(temp.Data[1][-1] - temp.Data[0][0], 2),
            # '%.2f%%' % (100 * (temp.Data[1][-1] - temp.Data[0][0])/temp.Data[0][0])]
            (temp.Data[1][-1] - temp.Data[0][0]) / temp.Data[0][0]]

def get_year(windcode):
    temp = w.wsd(windcode[1], "open,close", str(int(end_day[0:4])-1) + '-12-31', end_day)
    return windcode + [temp.Data[1][-1], round(temp.Data[1][-1] - temp.Data[0][0], 2),
            # '%.2f%%' % (100 * (temp.Data[1][-1] - temp.Data[0][0])/temp.Data[0][0])]
            (temp.Data[1][-1] - temp.Data[0][0]) / temp.Data[0][0]]


hk_index = [['恒生指数','HSI.HI'], ['—（恒生金融）','HSF.HI'], ['—（恒生工商）','HSCII.HI'],
            ['—（恒生地产）','HSPI.HI'], ['—（恒生公共事业）','HSUI.HI'],
            ['恒生国企指数','HSCEI.HI'], ['恒生红筹指数','HSCCI.HI'], ['恒生高股息率指数','HSHDYI.HI'],
            ['标普500', 'SPX.GI'],['纳斯达克指数','IXIC.GI'],['道琼斯工业指数','DJI.GI']]
world_index = [['恒生指数','HSI.HI'], ['恒生国企指数','HSCEI.HI'], ['恒生红筹指数','HSCCI.HI'],
               ['恒生高股息指数','HSHDYI.HI'], ['上证50','000016.SH'], ['沪深300','000300.SH'],
               ['中证500','000905.SH'], ['道琼斯工业指数','DJI.GI'],
               ['纳斯达克指数','IXIC.GI'], ['标普500','SPX.GI'], ['罗素2000','IWM.P']]
industry_index = [['金融','HSFSI.HI'], ['资讯科技','HSITSI.HI'], ['地产建筑','HSPCSI.HI'],
                  ['消费品制造','HSCGSI.HI'], ['工业','HSGSI.HI'], ['公共事业','HSUSI.HI'], ['电讯业','HSTSI.HI'],
                  ['消费者服务','HSCISV.HI'], ['能源','HSESI.HI'], ['原材料','HSCIMT.HI'], ['综合','HSCSI.HI']]

col_name = ['Name','Index','Close','Change', 'Return']
df = pd.concat([pd.DataFrame(map(get_price, hk_index),columns=col_name),
                             pd.DataFrame(['全球市场']+[None]*(len(col_name)-1),index=col_name).T,
                             pd.DataFrame(map(get_price, world_index),columns=col_name).sort_values(by='Return', ascending=False),
                             pd.DataFrame(['全球市场(年初至今)']+[None]*(len(col_name)-1),index=col_name).T,
                             pd.DataFrame(map(get_year, world_index),columns=col_name).sort_values(by='Return', ascending=False),
                             pd.DataFrame(['行业板块']+[None]*(len(col_name)-1),index=col_name).T,
                             pd.DataFrame(map(get_price, industry_index),columns=col_name).sort_values(by='Return',ascending=False),
                             pd.DataFrame(['行业板块(年初至今)'] + [None] * (len(col_name) - 1), index=col_name).T,
                             pd.DataFrame(map(get_year, industry_index), columns=col_name).sort_values(by='Return', ascending=False)])
df.columns = [start_day, end_day, '收盘', '涨跌', '涨跌幅']
df.to_csv('C:\\Users\\yz283\\Desktop\\df.csv', encoding='utf_8_sig')
print ('Done!')
