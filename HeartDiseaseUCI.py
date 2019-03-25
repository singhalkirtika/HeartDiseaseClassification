# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %matplotlib inline

# %%
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import os
import warnings
warnings.filterwarnings('ignore')

# %%
data = pd.read_csv("heart.csv")
data.head()

# %%
data.describe()

# %%
data.isnull().any()

# %%
plt.figure(figsize=(10,10))
sns.heatmap(data.corr(),annot=True,fmt='.1f')
plt.show()

# %%
sns.pairplot(data, hue="target", diag_kind = 'hist')
plt.show()

# %% [markdown]
# ### Age Analysis

# %%
age_min = list(sorted(set(data.age)))[0]
age_max = list(sorted(set(data.age)))[-1]
age_mean = (age_min+age_max)//2

print("Minimum age ", age_min)
print("Maximum age ",  age_max)
print("Mean age ", age_mean)

# %%
fig, ax1 = plt.subplots(1,1, figsize = (20,5))
bin_x = range(25,80,2)

ax1.hist(data.age.tolist(), bins = bin_x, rwidth=0.9)
ax1.set_xticks(range(25,80,2))
ax1.set_xlabel('Age',fontsize=15)
ax1.set_ylabel('Population Count',fontsize=15)
ax1.set_title('Total population distribution',fontsize=20)

# %%
sns.violinplot(data.age, palette="Set2", bw=.2, cut=1, linewidth=1)
plt.xticks(rotation=90)
plt.title("Age Rates")
plt.show()

# %%
x = data.groupby(['age','target']).agg({'sex':'count'})
y = data.groupby(['age']).agg({'sex':'count'})
z = (x.div(y, level='age') * 100)
q= 100 - z

fig, (ax1, ax2) = plt.subplots(1,2, figsize = (20,6))
plt.subplots_adjust(hspace = 0.5)

ax1.hist(data[data['target']==1].age.tolist(),bins=bin_x,rwidth=0.8)
ax1.set_xticks(range(25,80,2))
ax1.set_xlabel('Age Range',fontsize=15)
ax1.set_ylabel('Population Count',fontsize=15)
ax1.set_title('People suffering from heart disease',fontsize=20)

ax2.scatter(z.xs(1,level=1).reset_index().age,z.xs(1,level=1).reset_index().sex,s=(x.xs(1,level=1).sex)*30,edgecolors = 'r',c = 'yellow')
ax2.plot(z.xs(1,level=1).reset_index().age,z.xs(1,level=1).reset_index().sex)
ax2.set_xticks(range(25,80,2))
ax2.set_yticks(range(0,110,5))
ax2.set_xlabel('Age',fontsize=15)
ax2.set_ylabel('%',fontsize=15)
ax2.set_title('% of people with heart disease by age',fontsize=20)

plt.show()


# %%
young_ages = data[(data.age>=29)&(data.age<40)]
middle_ages = data[(data.age>=40)&(data.age<55)]
elderly_ages = data[(data.age>55)]
print('Young Ages :',len(young_ages))
print('Middle Ages :',len(middle_ages))
print('Elderly Ages :',len(elderly_ages))

# %%
sns.barplot(x=['young ages','middle ages','elderly ages'],y=[len(young_ages),len(middle_ages),len(elderly_ages)])
plt.xlabel('Age Range')
plt.ylabel('Age Counts')
plt.title('Ages State in Dataset')
plt.show()

# %%
data['ageRange'] = 0
youngAge_index = data[(data.age>=29)&(data.age<40)].index
middleAge_index = data[(data.age>=40)&(data.age<55)].index
elderlyAge_index = data[(data.age>55)].index

# %%
for index in elderlyAge_index:
    data.loc[index,'ageRange'] = 2
    
for index in middleAge_index:
    data.loc[index,'ageRange'] = 1

for index in youngAge_index:
    data.loc[index,'ageRange'] = 0

# %%
# Draw a categorical scatterplot to show each observation
sns.swarmplot(x="ageRange", y="age", hue='sex',  data=data)
plt.show()

# %% [markdown]
# #### 1. Majority of people suffering from heart disease lies between age 40 to 65
# #### 2. Proability of getting heart disease starts reduce significiently after age of 60
# #### 3. People from age 37 to 59 has highest chance of getting heart disease by volume

# %% [markdown]
# ### Gender Analysis

# %%
male_count = data.sex.value_counts().tolist()[0]
female_count = data.sex.value_counts().tolist()[1]

fig, (ax1,ax2) = plt.subplots(1,2,figsize = (12,5))
plt.subplots_adjust(wspace = 0.5)

ax1.bar(data.sex.unique(), data.sex.value_counts(),color = ['blue','red'],width = 0.8)
ax1.set_xticks(data.sex.unique())
ax1.set_xticklabels(('Male','Female'))

ax2.pie((male_count,female_count), labels = ('Male','Female'), autopct='%1.1f%%', shadow=True, startangle=90, explode=[0,0.2])

plt.show()

# %%
#Male State & target 1 & 0
male_target_on=len(data[(data.sex==1)&(data['target']==1)])
male_target_off=len(data[(data.sex==1)&(data['target']==0)])
####
sns.barplot(x=['Male Target On','Male Target Off'],y=[male_target_on,male_target_off])
plt.xlabel('Male and Target State')
plt.ylabel('Count')
plt.title('State of the Gender')
plt.show()

# %%
#Male State & target 1 & 0
female_target_on=len(data[(data.sex==0)&(data['target']==1)])
female_target_off=len(data[(data.sex==0)&(data['target']==0)])
####
sns.barplot(x=['Female Target On','Female Target Off'],y=[female_target_on,female_target_off])
plt.xlabel('Female and Target State')
plt.ylabel('Count')
plt.title('State of the Gender')
plt.show()

# %% [markdown]
# #### In this section, the rate of disease is seen less when the gender value is male. This is the result of an analysis for us.

# %%
# Plot miles per gallon against horsepower with other semantics
sns.regplot(x="trestbps", y="age", data=data)

# %% [markdown]
# ### Chest Pain Type Analysis

# %%
sns.countplot(data.cp)
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.title("Chest Pain Type vs Count State")
plt.show()

# %%
fig, axes = plt.subplots(2,2, figsize = (18,8))
plt.subplots_adjust(hspace = 0.5)

cp_zero_target_zero=len(data[(data.cp==0)&(data.target==0)])
cp_zero_target_one=len(data[(data.cp==0)&(data.target==1)])

axes[0,0].bar([0, 1], [cp_zero_target_zero, cp_zero_target_one])
axes[0,0].set_xticks([0,1])
axes[0,0].set_xlabel('Target')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('cp = 0')

cp_one_target_zero=len(data[(data.cp==1)&(data.target==0)])
cp_one_target_one=len(data[(data.cp==1)&(data.target==1)])
axes[0,1].bar([0, 1], [cp_one_target_zero, cp_one_target_one])
axes[0,1].set_xticks([0,1])
axes[0,1].set_xlabel('Target')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('cp = 1')

cp_two_target_zero=len(data[(data.cp==2)&(data.target==0)])
cp_two_target_one=len(data[(data.cp==2)&(data.target==1)])
axes[1,0].bar([0, 1], [cp_two_target_zero, cp_two_target_one])
axes[1,0].set_xticks([0,1])
axes[1,0].set_xlabel('Target')
axes[1,0].set_ylabel('Count')
axes[1,0].set_title('cp = 2')

cp_three_target_zero=len(data[(data.cp==3)&(data.target==0)])
cp_three_target_one=len(data[(data.cp==3)&(data.target==1)])
axes[1,1].bar([0, 1], [cp_three_target_zero, cp_three_target_one])
axes[1,1].set_xticks([0,1])
axes[1,1].set_xlabel('Target')
axes[1,1].set_ylabel('Count')
axes[1,1].set_title('cp = 3')

plt.show()

# %% [markdown]
# #### As a result of the above analyzes, it can be seen cases with cp=0 are less common with heart disease. But on the other hand, there are problems in all cases of chest pain, such as 1,2,3.

# %% [markdown]
# ### Thalach Analysis

# %%
fig, ax1 = plt.subplots(1,1, figsize = (18,5))
bin_x = range(70,210,2)

sns.barplot(x=data.thalach.value_counts().index,y=data.thalach.value_counts().values)
plt.xlabel('Thalach')
plt.ylabel('Count')
plt.title('Thalach Counts')
plt.xticks(rotation=90)
plt.show()

# %%
age_unique = sorted(data.age.unique())
age_thalach_values = data.groupby('age')['thalach'].count().values
mean_thalach=[]
for i,age in enumerate(age_unique):
    mean_thalach.append(sum(data[data['age']==age].thalach)/age_thalach_values[i])
    
#data_sorted=data.sort_values(by='Age',ascending=True)
plt.figure(figsize=(10,5))
sns.pointplot(x=age_unique,y=mean_thalach,color='red',alpha=0.8)
plt.xlabel('Age',fontsize = 15,color='blue')
plt.xticks(rotation=45)
plt.ylabel('Thalach',fontsize = 15,color='blue')
plt.title('Age vs Thalach',fontsize = 15,color='blue')
plt.grid()
plt.show()

# %%
age_range_thalach=data.groupby('ageRange')['thalach'].mean()
sns.barplot(x=age_range_thalach.index,y=age_range_thalach.values)
plt.xlabel('Age Range Values')
plt.ylabel('Maximum Thalach By Age Range')
plt.title('illustration of the thalach to the age range')
plt.show()
#As shown in this graph, this rate decreases as the heart rate is faster and in old age areas.

# %%
cp_thalach=data.groupby('cp')['thalach'].mean()

sns.barplot(x=cp_thalach.index,y=cp_thalach.values)
plt.xlabel('Degree of Chest Pain (Cp)')
plt.ylabel('Maximum Thalach By Cp Values')
plt.title('Illustration of thalach to degree of chest pain')
plt.show()
#As seen in this graph, it is seen that the heart rate is less 
#when the chest pain is low. But in cases where chest pain is 
#1, 2 or 3 it is observed that the area is more

# %% [markdown]
# ### Thal Analysis

# %%
sns.countplot(data.thal)
plt.show()

# %%
#Target 1
a=len(data[(data['target']==1)&(data['thal']==0)])
b=len(data[(data['target']==1)&(data['thal']==1)])
c=len(data[(data['target']==1)&(data['thal']==2)])
d=len(data[(data['target']==1)&(data['thal']==3)])
print('Target 1 Thal 0: ',a)
print('Target 1 Thal 1: ',b)
print('Target 1 Thal 2: ',c)
print('Target 1 Thal 3: ',d)

#so,Apparently, there is a rate at Thal 2.Now, draw graph
print('*'*50)

#Target 0
e=len(data[(data['target']==0)&(data['thal']==0)])
f=len(data[(data['target']==0)&(data['thal']==1)])
g=len(data[(data['target']==0)&(data['thal']==2)])
h=len(data[(data['target']==0)&(data['thal']==3)])
print('Target 0 Thal 0: ',e)
print('Target 0 Thal 1: ',f)
print('Target 0 Thal 2: ',g)
print('Target 0 Thal 3: ',h)


# %%
f,ax = plt.subplots(figsize=(7,7))
sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,6,130,28],color='green',alpha=0.5,label='Target 1 Thal State')
sns.barplot(y=['T 1&0 Th 0','T 1&0 Th 1','T 1&0 Th 2','Ta 1&0 Th 3'],x=[1,12,36,89],color='red',alpha=0.7,label='Target 0 Thal State')
ax.legend(loc='lower right',frameon=True)
ax.set(xlabel='Target State and Thal Counter',ylabel='Target State and Thal State',title='Target VS Thal')
plt.xticks(rotation=90)
plt.show()
#so, there has been a very nice graphic display. This is the situation that best describes the situation.

# %% [markdown]
# ### Target Analysis

# %%
sns.countplot(data.target)
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Target Counter 1 & 0')
plt.show()

# %%
g = sns.catplot(x="ageRange", y="chol", hue="sex",
                 data=data, kind="bar", 
                 height=4, aspect=.7)
plt.show()

# %%
male_young_t_1=data[(data['sex']==1)&(data['ageRange']==0)&(data['target']==1)]
male_middle_t_1=data[(data['sex']==1)&(data['ageRange']==1)&(data['target']==1)]
male_elderly_t_1=data[(data['sex']==1)&(data['ageRange']==2)&(data['target']==1)]
print(len(male_young_t_1))
print(len(male_middle_t_1))
print(len(male_elderly_t_1))

f,ax1=plt.subplots(figsize=(20,8))
sns.pointplot(x=np.arange(len(male_young_t_1)),y=male_young_t_1.trestbps,color='lime',alpha=0.8,label='Young')
sns.pointplot(x=np.arange(len(male_middle_t_1)),y=male_middle_t_1.trestbps,color='black',alpha=0.8,label='Middle')
sns.pointplot(x=np.arange(len(male_elderly_t_1)),y=male_elderly_t_1.trestbps,color='red',alpha=0.8,label='Elderly')
plt.xlabel('Range',fontsize = 15,color='blue')
plt.xticks(rotation=90)
plt.legend(loc='upper right',frameon=True)
plt.ylabel('Trestbps',fontsize = 15,color='blue')
plt.title('Age Range Values vs Trestbps',fontsize = 20,color='blue')
plt.grid()
plt.show()

# %%
for i,col in enumerate(data.columns.values):
    plt.subplot(5,3,i+1)
    plt.scatter([i for i in range(303)],data[col].values.tolist())
    plt.title(col)
    fig,ax=plt.gcf(),plt.gca()
    fig.set_size_inches(10,10)
    plt.tight_layout()
plt.show()

# %% [markdown]
# ### Model Training

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,RobustScaler, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# %%
num_columns =  ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

print('Columns with continous data = ',num_columns)

import scipy.stats as stats

fig, axes = plt.subplots(2,2, figsize = (15,12))
plt.subplots_adjust(hspace = 0.2)

h= np.sort(data.thalach)
fit = stats.norm.pdf(h, np.mean(h), np.std(h)) 
axes[0,0].plot(h,fit,'--')
axes[0,0].hist(h,density=True) 
axes[0,0].set_title("thalach")
axes[0,0].set_ylabel('Density')

h2= np.sort(data.trestbps)
fit2 = stats.norm.pdf(h2, np.mean(h2), np.std(h2)) 
axes[0,1].plot(h2,fit2,'--')
axes[0,1].hist(h2,density=True) 
axes[0,1].set_title("trestbps")
axes[0,1].set_ylabel('Density')

h3= np.sort(data.chol)
fit3 = stats.norm.pdf(h3, np.mean(h3), np.std(h3)) 
axes[1,0].plot(h3,fit3,'--')
axes[1,0].hist(h3,density=True) 
axes[1,0].set_title("chol")
axes[1,0].set_ylabel('Density')

h4= np.sort(data.oldpeak)
fit4 = stats.norm.pdf(h4, np.mean(h4), np.std(h4)) 
axes[1,1].plot(h4,fit4,'--')
axes[1,1].hist(h4,density=True) 
axes[1,1].set_title("oldpeak")
axes[1,1].set_ylabel('Density')

plt.show()

print(r"Scaling them using MinMax Scaler")

# %%
mm = MinMaxScaler()

num_data = data[num_columns]
num_data_tf = mm.fit_transform(num_data)

# %% [markdown]
# ### One Hot encoding all catagorical columns

# %%
ohe = OneHotEncoder()
cat_columns = [clm_name for clm_name in data.columns if clm_name not in num_columns]

cat_columns.remove('target')
cat_data = data[cat_columns]
cat_data_tf = ohe.fit_transform(cat_data).toarray()

data_tf = np.hstack([num_data_tf,cat_data_tf])
data_tf_df = pd.DataFrame(data_tf)

# %%
X = data.drop('target',axis=1)
Y = data['target']

X_tf = data_tf_df

# %%
print("Split the data into train and test with unscaled data:")
trainX,testX,trainY,testY = train_test_split(X,Y,test_size = 0.3,random_state = None)
print("trainX,testX,trainY,testY")
print("Split the data into train and test with scaled data:")
trainX_tf,testX_tf,trainY_tf,testY_tf = train_test_split(X_tf,Y,test_size = 0.3,random_state = None)
print("trainX_tf,testX_tf,trainY_tf,testY_tf")

# %% [markdown]
# ### Using Random Forest Classifier with unscalled data and Grid Search CV

# %%
from sklearn.tree import export_graphviz #plot tree

# %%
rf = RandomForestClassifier()

# %%
#Using grid search to get best params for Randomforest
params = {
    'n_estimators':[10,50,100,150,200,250],
    'random_state': [10,5,15,20,50]
         }
gs = GridSearchCV(rf, param_grid=params, cv=5, n_jobs=-1)
gs.fit(trainX,trainY)

# %%
n_est = []
rnd_sta = []
score = []
rand_state_list =  [5,10,15,20,50]
for x in range(len(gs.cv_results_['params'])):
    n_est.append(gs.cv_results_['params'][x]['n_estimators'])
    rnd_sta.append(gs.cv_results_['params'][x]['random_state'])
    score.append(gs.cv_results_['mean_test_score'][x])

grid_frame = pd.DataFrame()
grid_frame['n_est'] = n_est
grid_frame['rnd_sta'] = rnd_sta
grid_frame['score'] = score

grid_frame[grid_frame['rnd_sta'] == 10]

plt.figure(figsize=(10,6))

for value in rand_state_list:
    plt.plot(grid_frame[grid_frame['rnd_sta'] == value].n_est,grid_frame[grid_frame['rnd_sta'] == value].score,'-o',label = 'random_state = value')

plt.title('Mean Score with different params')
plt.grid(True)
plt.legend()
plt.show()

# %%
# Grid Search Score with test Data
print("Grid search score with random forest classifier = ",gs.score(testX,testY)*100)

# %%
# Using grid search to get best params for Randomforest . Now with scaled data (StandardScaler)
params = {
    'n_estimators':[10,50,100,150,200,250,300],
    'random_state': [10,5,15,20,50]
         }
gs = GridSearchCV(rf, param_grid=params, cv=10, n_jobs=-1)
gs.fit(trainX_tf,trainY_tf)

# %%
n_est = []
rnd_sta = []
score = []
for x in range(len(gs.cv_results_['params'])):
    n_est.append(gs.cv_results_['params'][x]['n_estimators'])
    rnd_sta.append(gs.cv_results_['params'][x]['random_state'])
    score.append(gs.cv_results_['mean_test_score'][x])

grid_frame = pd.DataFrame()
grid_frame['n_est'] = n_est
grid_frame['rnd_sta'] = rnd_sta
grid_frame['score'] = score

grid_frame[grid_frame['rnd_sta'] == 10]

plt.figure(figsize=(10,6))

for value in rand_state_list:
    plt.plot(grid_frame[grid_frame['rnd_sta'] == value].n_est,grid_frame[grid_frame['rnd_sta'] == value].score,'-o',label = 'random_state = value')

plt.title('Mean Score with different params')
plt.grid(True)
plt.legend()
plt.show()

# %%
# Grid Search Score wih scaled test data 
print("Grid search score with random forest classifier (Scaled Data)= ",gs.score(testX_tf,testY_tf)*100)

# %%
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier,RandomForestClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# %%
models = [SVC(kernel='linear',C =100),
          SGDClassifier(max_iter=1000,tol=0.003),
          DecisionTreeClassifier(),
          ExtraTreeClassifier(),
          AdaBoostClassifier(), 
          BaggingClassifier(), 
          GradientBoostingClassifier(),
          RandomForestClassifier(n_estimators=n_est,random_state=rnd_sta),
          GaussianNB(),
          KNeighborsClassifier(), 
          LogisticRegression(max_iter=1000,solver='lbfgs')]

modelnames = ['SVC',
              'SGDClassifier',
              'DecisionTreeClassifier',
              'ExtraTreeClassifier',
              'AdaBoostClassifier', 
              'BaggingClassifier', 
              'GradientBoostingClassifier',
              'RandomForestClassifier',
              'GaussianNB',
              'KNeighborsClassifier', 
              'LogisticRegression']

# %%
for index,model in enumerate(models):
    try:
        model.fit(trainX_tf,trainY_tf)
        print(modelnames[index],"Accuracy =",round(model.score(testX_tf,testY_tf)*100,2),"%")
    except:
        print("Skipped",modelnames[index])

# %%
#  base_estimator=DecisionTreeClassifier
ab = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(),n_estimators=1000)
ab.fit(trainX_tf,trainY_tf)
print('AdaBoost Accuracy with Decision Tree (Scaled Data)= ',(ab.score(testX_tf,testY_tf)*100))

# %%
#  base_estimator=RandomForest
ab = AdaBoostClassifier(base_estimator=RandomForestClassifier(n_estimators=1000,random_state=10),n_estimators=1000)
ab.fit(trainX_tf,trainY_tf)
print('AdaBoost Accuracy with Random Forest (Scaled Data)= ',(ab.score(testX_tf,testY_tf)*100))

# %%
#  base_estimator=LogisticRegression
ab = AdaBoostClassifier(base_estimator=LogisticRegression(max_iter=1000,solver = 'lbfgs'),n_estimators=1000)
ab.fit(trainX_tf,trainY_tf)
print('AdaBoost Accuracy with Logistic Reg (Scaled Data)= ',(ab.score(testX_tf,testY_tf)*100))

# %%
#  base_estimator=SGDClassifier
ab = AdaBoostClassifier(algorithm='SAMME',base_estimator=SGDClassifier(max_iter=1000, tol = 0.001),n_estimators=1000)
ab.fit(trainX_tf,trainY_tf)
print('AdaBoost Accuracy with SGDClassifier (Scaled Data)= ',(ab.score(testX_tf,testY_tf)*100))

# %%
#  base_estimator=SVC
ab = AdaBoostClassifier(algorithm='SAMME',base_estimator=SVC(kernel='linear',C = 1000, gamma=1),n_estimators=1000)
ab.fit(trainX_tf,trainY_tf)
print('AdaBoost Accuracy with SVC = (Scaled Data)',(ab.score(testX_tf,testY_tf)*100))
