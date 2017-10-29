#-*- coding: UTF-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=20)
# http://blog.csdn.net/xiangqianzsh/article/details/44929529
dat = pd.read_csv('C:\\Users\\yz283\\Desktop\\sh50.csv')
dat.columns = ['Date','q','p','score','return']
dat = dat[['Date','score','return']]
dat['return'] = dat['return']*100

sco = [0,1,2,3,7,8,9,10]
tt = pd.DataFrame()
for i in sco:
    temp = dat[dat.score == i]['return']
    temp.index = range(len(temp))
    if i < 5.5:
        ttemp = round(len(temp[temp < 0])/float(len(temp))*100, 0)
    else:
        ttemp = round(len(temp[temp > 0]) / float(len(temp))*100, 0)
    temp.name = str(i)+'分'+'('+str(len(temp))+')'+',胜率'+"%.0f%%" %ttemp
    tt = pd.concat([tt, temp], axis = 1)
t= tt.boxplot()
for label in t.get_xticklabels():
    label.set_fontproperties(font)
plt.show()