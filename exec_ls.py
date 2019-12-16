
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 00:30:02 2019

@author: Win10
"""
import numpy as np
import os
os.getcwd()

# 데이터 읽기
import pandas as pd
mdat = pd.read_csv('./LS미래원 DT교육용(전선CCV)_데이터샘플_v1.0.csv')
print(mdat.shape)

mdat.head() #mdat[:5]
mdat.columns
#mdat3 = mdat.set_index('DateTime')

#데이터 특성 보기
##boxplot
##barplot
##histogram

#한글 폰트 설정
import platform
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False
elif platform.system() == 'Windows':
    path = "c:\Windows\Fonts\malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
elif platform.system() == 'Linux':
    rc('font', family='NanumBarunGothic')
else:
    print("Error...")
    
## 윈도에서는 
## dir c:\Windows\Fonts\m*
## 로 폰트 확인
mdat[['편심률_계측기1(y1)']].boxplot(figsize=(10,6))
mdat[['편심률_계측기2(y2)']].boxplot(figsize=(10,6))
# => y1과 y2비교시 y2의 분포는 30전후로 상승함
mdat[['편심률_계측기1(y1)','편심률_계측기2(y2)']].hist()

# 산점도
import matplotlib.pyplot as plt
plt.figure(figsize=(7,6))
plt.scatter(mdat['편심률_계측기1(y1)'], mdat['편심률_계측기2(y2)'], s=1, c='k', alpha=0.1)
plt.xlim(10,40); plt.ylim(10,40)
#plt.xscale('log'); plt.yscale('log')
plt.xlabel('eccentricity_1', fontsize=12); plt.ylabel('eccentricity_2', fontsize=12)
#plt.plot(x, y, '--')
abline_values = [1 * i + 0 for i in [10,20,30,40]]
plt.plot([10,20,30,40], abline_values, 'b', alpha=0.3)
slope, intercept = np.polyfit(mdat['편심률_계측기1(y1)'], mdat['편심률_계측기2(y2)'], 1)
slope; intercept

plt.figure(figsize=(12,6))
plt.plot(mdat['편심률_계측기1(y1)'], c='k', ls=':', lw=0.1)
plt.plot(mdat['편심률_계측기2(y2)'], c='r', ls='-.', lw=0.1)
plt.show()
# Create a list of values in the best fit line

plt.figure(figsize=(12,6))
plt.plot(mdat['편심률_계측기2(y2)']-mdat['편심률_계측기1(y1)'], c='k', ls=':', lw=0.2)
plt.show()

#plt.plot(x, y, '--')
abline_values = [1 * i + 0 for i in [10,20,30,40]]
plt.plot([10,20,30,40], abline_values, 'b', alpha=0.3)
slope, intercept = np.polyfit(mdat['편심률_계측기1(y1)'], mdat['편심률_계측기2(y2)'], 1)
slope
intercept

plt.figure(figsize=(7,6))
plt.scatter(mdat['압출기 수지온도(x2)'], mdat['편심률_계측기1(y1)'])
plt.xlabel('압출기 수지온도(x2)')
plt.ylabel('eccentricity_1')

plt.figure(figsize=(7,6))
plt.scatter(mdat['압출기 수지온도(x2)'], mdat['편심률_계측기2(y2)'], s=1, c='k', alpha=0.1)
plt.xlabel('압출기 수지온도(x2)'); plt.ylabel('eccentricity_2')
slope, intercept = np.polyfit(mdat['압출기 수지온도(x2)'], mdat['편심률_계측기2(y2)'], 1)
slope; intercept
abline_values = [slope * i + intercept for i in [0.75,1,1.5,2,2.25]]
plt.plot([0.75,1,1.5,2,2.25], abline_values, 'b', alpha=0.3)

plt.figure(figsize=(7,6))
plt.scatter(mdat['압출기 압력(x4)'], mdat['편심률_계측기2(y2)'], s=1, c='k', alpha=0.1)
plt.ylabel('편심률_계측기2(y2)'); plt.xlabel('압출기 압력(x4)')
slope, intercept = np.polyfit(mdat['압출기 압력(x4)'], mdat['편심률_계측기2(y2)'], 1)
slope; intercept
abline_values = [slope * i + intercept for i in [np.min(mdat['압출기 압력(x4)']), np.max(mdat['압출기 압력(x4)'])]]
plt.plot([np.min(mdat['압출기 압력(x4)']), np.max(mdat['압출기 압력(x4)'])], abline_values, 'b', alpha=0.3)

"""
#'y1 vs. y2
# '도체 예열 온도(x1)'
fig = plt.figure()
g1 = ggplot(mdat, aes(x='편심률_계측기1(y1)', y='편심률_계측기2(y2)', color='도체 예열 온도(x1)')) 
g1 + geom_point(alpha=0.2) + ggtitle('y1 vs. y2(color=x1)')

# 압출기 압력(x4)
from plotnine import *
fig = plt.figure()
g4 = ggplot(mdat, aes(x='편심률_계측기1(y1)', y='편심률_계측기2(y2)', color='압출기 압력(x4)')) 
g4 + geom_point(alpha=0.02) + ggtitle('y1 vs. y2(color=x4)') #+ facet_grid()

# '압출기 수지온도(x2)'
fig = plt.figure()
g2 = ggplot(mdat, aes(x='편심률_계측기1(y1)', y='편심률_계측기2(y2)', color='압출기 수지온도(x2)')) 
g2 + geom_point(alpha=0.2) + ggtitle('y1 vs. y2(color=x2)')

# '압출기 수지온도(x2)'
fig = plt.figure()
g2 = ggplot(mdat, aes(x='압출기 수지온도(x2)', y='편심률_계측기2(y2)', color='압출기 수지온도(x2)')) 
g2 + geom_point(alpha=0.2) + ggtitle('y1 vs. y2(color=x2)')

# '압출기 모터RPM(x3)'
fig = plt.figure()
g3 = ggplot(mdat, aes(x='압출기 모터RPM(x3)', y='편심률_계측기2(y2)')) 
g3 + geom_point(alpha=0.2) + ggtitle('y1 vs. y2(color=x2)')

fig = plt.figure()
g3 = ggplot(mdat, aes(x='압출기 모터RPM(x3)', y='편심률_계측기2(y2)')) 
g3 + geom_point(alpha=0.2) + ggtitle('y1 vs. y2(color=x2)')

g4 = ggplot(mdat, aes(x='압출기 압력(x4)', y='편심률_계측기2(y2)')) 
g4 + geom_point(alpha=0.2) + ggtitle('y1 vs. y2(color=x2)')

g5 = ggplot(mdat, aes(x='압출기 실린더 온도(x5)', y='편심률_계측기2(y2)')) 
g5 + geom_point(alpha=0.2) + ggtitle('y1 vs. y2(color=x2)')

g6 = ggplot(mdat, aes(x='냉각기 입구 온도(x6)', y='편심률_계측기2(y2)')) 
g6 + geom_point(alpha=0.2) + ggtitle('y1 vs. y2(color=x2)')

g7 = ggplot(mdat, aes(x='냉각기 출구 온도(x7)', y='편심률_계측기2(y2)')) 
g7 + geom_point(alpha=0.2) + ggtitle('x7 vs. y2')

g8 = ggplot(mdat, aes(x='가교관 온도(x8)', y='편심률_계측기2(y2)')) 
g8 + geom_point(alpha=0.2) + ggtitle('x8 vs. y2')

g9 = ggplot(mdat, aes(x='조장 길이(x9)', y='편심률_계측기2(y2)')) 
g9 + geom_point(alpha=0.2) + ggtitle('x9 vs. y2')

import seaborn as sns
sns.pairplot(mdat, diag_kind='hist')
plt.show()
"""

plt.figure(figsize=(7,6))
plt.scatter(mdat['압출기 수지온도(x2)'], mdat['편심률_계측기2(y2)'], s=1, c='k', alpha=0.1)
plt.xlabel('압출기 수지온도(x2)'); plt.ylabel('eccentricity_2')
slope, intercept = np.polyfit(mdat['압출기 수지온도(x2)'], mdat['편심률_계측기2(y2)'], 1)
slope; intercept
abline_values = [slope * i + intercept for i in [0.75,1,1.5,2,2.25]]
plt.plot([0.75,1,1.5,2,2.25], abline_values, 'b', alpha=0.3)

plt.figure(figsize=(7,6))
plt.scatter(mdat['압출기 수지온도(x2)']-mdat['편심률_계측기1(y1)'], mdat['편심률_계측기2(y2)'])
plt.xlabel('압출기 수지온도(x2)')
plt.ylabel('차이')

"""
for n in range(n_samples):
    plt.text(power['서비스업'][n]*1.02, power['제조업'][n]*0.99, power.index[n])
power = power.drop(['경기', '서울'])
n_samples = power.shape[0]
"""

####################################
####################################
# 결측값 확인하기
##칼럼별 결측값 개수 구하기 : 
mdat.isnull().sum()
pos = np.array(mdat['조장 길이(x9)'].isnull()) == True
xx = mdat[['조장 길이(x9)']].isnull().loc[mdat['조장 길이(x9)'].isnull() == True]
list(xx.index.values)

## 행단위 결측값
#mdat['NaN_cnt'] = mdat.isnull().sum(1)
mdat.isnull()
###
## 3개의 자료는 제외하기로 함
mdat2 = mdat.dropna()
mdat2.shape

##
mdat2.columns = ['DateTime', 'y1', 'y2', 'x1','x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']
mdat2[['y2']].boxplot()
mdat2.quantile([0.25,0.5,0.75])
mdat2[['y2']].quantile(0.1)
xx = mdat2[['x9']].loc[mdat2['y2'] <25]
len(list(xx.index.values))
mdat21 = mdat2[mdat2['y2'] > 25]
mdat21[['x1']].quantile([0.25,0.75])
mdat21[['x1']].boxplot()
#(99.7-97.8)*1.5 => 2.85
#97.8-2.85 => 94.95
mdat21_d = mdat21.set_index('DateTime')
mdat22 = mdat21[mdat21['x1'] > 94.85]
"""
import seaborn as sns
sns.pairplot(mdat22, diag_kind='hist')
plt.show()
"""


"""
#plt.xticks(fontsize=20, fontweight='bold',rotation=90)
#plt.yticks(fontsize=20, fontweight='bold')
#plt.xlabel('Dates',fontsize=20, fontweight='bold')
#plt.ylabel('Total Count',fontsize=20, fontweight='bold')
#plt.title('Counts per time',fontsize=20, fontweight='bold')
"""
mdat22_d = mdat22.set_index('DateTime')
mdat22_d['y2']-mdat22_d['y1']
mdat22_d = mdat22_d.assign(y21=mdat22_d.y2-mdat22_d.y1)
mdat22_d.columns

"""
import seaborn as sns
sns.pairplot(mdat22_d, diag_kind='hist')
plt.show()
"""
# 덴드로그램
#mdat2 = mdat.drop(['서울', '경기'])
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import distance
from sklearn.cluster import KMeans

mdat4 = mdat22_d.T
mdat4.index = ['편심률_계측기1(y1)', '편심률_계측기2(y2)', '도체 예열 온도(x1)',
       '압출기 수지온도(x2)', '압출기 모터RPM(x3)', '압출기 압력(x4)', '압출기 실린더 온도(x5)',
       '냉각기 입구 온도(x6)', '냉각기 출구 온도(x7)', '가교관 온도(x8)', '조장 길이(x9)', 'y2-y1']
Z = linkage(mdat4, metric='correlation', method='complete')
# 유클리드 거리를 이용해 Linkage Matrix를 생성
Z

plt.figure(figsize=(10, 5))
plt.title('dendrogram')
dendrogram(Z, labels=mdat4.index)
plt.xticks(fontsize=12, fontweight='bold',rotation=90)
plt.show()

# heatmap by plt.pcolor()
#plt.pcolor(df)
#plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
#plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
#plt.title('Heatmap by plt.pcolor()', fontsize=20)
#plt.xlabel('Year', fontsize=14)
#plt.ylabel('Month', fontsize=14)
#plt.colorbar()
#plt.show()

import seaborn as sns
sns.heatmap(Z)

plt.figure(figsize=(8,8))
mdat22_d1 = mdat22_d.copy()
mdat22_d1.columns = ['편심률_계측기1(y1)', '편심률_계측기2(y2)', '도체 예열 온도(x1)',
       '압출기 수지온도(x2)', '압출기 모터RPM(x3)', '압출기 압력(x4)', '압출기 실린더 온도(x5)',
       '냉각기 입구 온도(x6)', '냉각기 출구 온도(x7)', '가교관 온도(x8)', '조장 길이(x9)', 'y2-y1']
sns.heatmap(data = mdat22_d1.corr(), annot=True, fmt = '.2f', linewidths=.5, cmap='Blues')


"""
#클러스터링 (KMeans)
k = KMeans(n_clusters= 4).fit(mdat3)
k.labels_

mdat3['cluster'] = k.labels_

power.drop('클러스터', axis = 1, inplace=True)
power

centers = k.cluster_centers_
centers
"""

# graph
from matplotlib import pyplot as plt

temp1 = mdat22_d['y1'] # y2
temp2 = mdat22_d['y2'] # y2
plt.plot(range(len(temp1)), temp1, alpha=0.4)
# 10초당 측정 => 처음 10분간의 자료
plt.plot(range(600), temp1[:600], alpha=0.4, c='b')
plt.plot(range(600), temp2[:600], alpha=0.4, c='r')

# learning
float_data = mdat22_d.to_numpy()
## 데이터 준비
mm = float_data[:18000].mean(axis=0)
float_data -= mm
std = float_data[:18000].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1] #y2
        yield samples, targets

lookback = 720 # 10초당 1 720*10/3600 =2시간 
step = 6
delay = 120   # 10초당 1개 1200/3600 = 20분
batch_size = 100        

np.random.seed(1)
train_gen = generator(float_data, lookback=lookback, delay=delay, min_index=0,
                      max_index=18000, shuffle=True,
                      step=step, batch_size=batch_size)
val_gen = generator(float_data, lookback=lookback, delay=delay, min_index=18001, 
                    max_index=21000, step=step, batch_size=batch_size)
test_gen = generator(float_data, lookback=lookback, delay=delay, min_index=21001,
                     max_index=None, step=step, batch_size=batch_size)

# 전체 검증 세트를 순회하기 위해 val_gen에서 추출할 횟수
val_steps = (20000 - 15001 - lookback) // batch_size
# 전체 테스트 세트를 순회하기 위해 test_gen에서 추출할 횟수
test_steps = (len(float_data) - 20001 - lookback) // batch_size
import sys
def evaluate_naive_method():
    batch_maes = []
    #for step in range(val_steps):
    for step in range(1):
        samples, targets = next(val_gen)
        print(type(samples))
        print(samples.shape)
        #print(type(targets))
        #print(targets.shape)
        preds = samples[:, -1, 1]
        #print(preds)
        #sys.exit(-1) 
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    
    print(np.mean(batch_maes))
    return batch_maes[0]

mse_base = evaluate_naive_method()
mse_base*std[1]

# model1: regression
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape \
                         = (lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.summary()

np.random.seed(2) 
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)

def trajector(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

import matplotlib.pyplot as plt
trajector(history)

# model2
model1 = Sequential()
model1.add(layers.GRU(32, input_shape \
                         = (None, float_data.shape[-1])))
model1.add(layers.Dense(1))
model1.summary()

np.random.seed(2) 
model1.compile(optimizer=RMSprop(), loss='mae')
history1 = model1.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
trajector(history1)

# model3: GRU with dropout
model2 = Sequential()
model2.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, \
                      input_shape = (None, float_data.shape[-1])))
model2.add(layers.Dense(1))
model2.summary()

np.random.seed(2) 
model2.compile(optimizer=RMSprop(), loss='mae')
history2 = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
trajector(history2)














#
#power1
#power1.boxplot(figsize=(10,6))
#plt.xticks(rotation=60)  































#