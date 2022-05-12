import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

df = pd.read_csv("C:/Users/21055/Downloads/fuel consumption of car fleet.txt", sep='\s+')
df.head()
sns.pairplot(df)
plt.savefig('/Users/21055/Downloads//test1.jpg')
plt.clf()

sns.heatmap(df.corr(), annot=True, linewidths=0.3)
# heatmap 热力图
# df.corr() 返回改数据类型的相关系数矩阵（即每两个类型直接的相关性）
plt.savefig('/Users/21055/Downloads//test2.jpg')
plt.show()


slr = LinearRegression()    # 线性拟合参数
x = df.drop(['Fuel_consumption_in_l'], axis=1)   # 去掉四列数据中的Fuel_consumption_in_l
y = df['Fuel_consumption_in_l']
slr.fit(x, y)     # 线性拟合输入参数
print('Slope 1: %.3f' %slr.coef_[0])    # 线性拟合输出参数
print('Slope 2: %.3f' %slr.coef_[1])
print('Slope 3: %.3f' %slr.coef_[2])
print('Intercept: %.3f' %slr.intercept_)   # 截距


