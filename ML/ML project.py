#!/usr/bin/env python
# coding: utf-8

# # Отток клиентов

# Нужно спрогнозировать, уйдёт клиент из банка в ближайшее время или нет. Вам предоставлены исторические данные о поведении клиентов и расторжении договоров с банком. 
# 
# Источник данных: [https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling](https://www.kaggle.com/barelydedicated/bank-customer-churn-modeling)

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


# ## Подготовка данных

# Для начала откроем файл и ознакомимся с общей информацией.

# In[2]:


df = pd.read_csv('C:/Users/123/Churn_Modelling.csv')


# In[3]:


display(df.head(10));
df.info()


# Мы будем иметь дело с 10000 клиентами. Необходимые признаки присутствуют, целевой признак тоже, что облегчит нам работу.

# Для удобства работы приведем названия столбцов к нижнему регистру.

# In[4]:


df.columns = df.columns.str.lower()


# Для начала посмотрим какую часть от общего объема данных составляют пропуски в столбце `Tenure(сколько лет человек является клиентом банка)`. Если менее 10% - целесообразно будет их удалить.

# In[5]:


df['tenure'].isna().sum() / len(df)


# In[6]:


df = df.dropna().reset_index(drop=True)
df.shape


# Фамилии клиентов, их ID и номер строки никак не влияют на целевой признак, можно удалить данные столбцы для упращения прямого кодирования.

# In[7]:


df = df.drop(['customerid', 'rownumber', 'surname'], axis=1)
df.head(10)


# Далее воспользуемся техникой прямого кодирования `One-Hot Encoding` для создания фиктивных переменных и переноса всех данных таблицы в численный вид.

# In[8]:


df_ohe = pd.get_dummies(df, drop_first=True)


# In[9]:


pd.set_option('display.max_columns', None)
df_ohe.head(10)


# Нужно разбить данные на 3 выборки
# - обучающая выборка
# - валидационная выборка
# - тестовая выборка

# In[10]:


features = df_ohe.drop(['exited'], axis=1)
target = df_ohe['exited']

df_train, df_valid_and_test = train_test_split(df_ohe, test_size=0.4, random_state=12345)
df_valid, df_test = train_test_split(df_valid_and_test, test_size=0.50, random_state=12345)

features_train = df_train.drop(['exited'], axis=1)
target_train = df_train['exited']

features_valid = df_valid.drop(['exited'], axis=1)
target_valid = df_valid['exited']

features_test = df_test.drop(['exited'], axis=1)
target_test = df_test['exited']

print(features.shape)
print(target.shape)
print()
print('Размеры обучающей выборки:')
print(features_train.shape)
print(target_train.shape)
print()
print('Размеры валидационной выборки:')
print(features_valid.shape)
print(target_valid.shape)
print()
print('Размеры тестовой выборки:')
print(features_test.shape)
print(target_test.shape)


# Мы получили 3 выборки в пропорциях 60/20/20. Обучающая выборка забрала 60% данных, тестовая и валидационная поделили оставшиеся 40%.

# ## Исследование задачи

# Для начала исследуем баланс классов.

# In[11]:


display(df['exited'].loc[df['exited'] == 1].count() / len(df))
display(df['exited'].loc[df['exited'] == 0].count() / len(df))


# 20% от общего числа клиентов прекратили пользоваться банком. Дисбаланс классов на лицо в соотношении 20/80.

# Обучим модель не взирая на дисбаланс классов. Метрика `accuracy_score` НЕ ПОКАЗАТЕЛЬНА для оценки моделей (в задаче классификации) построенных на данных с дисбалансом классов. Используем f1-меру.

# In[12]:


model = DecisionTreeRegressor(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

f1_score(target_valid, predicted_valid)


# In[13]:


model = RandomForestClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

f1_score(target_valid, predicted_valid)


# Значение F1-меры 0.589 - хотелось бы лучше. Нужно поработать с дисбалансом.

# ## Борьба с дисбалансом

# Постараемся подобрать наилучшие гиперпараметры для моделей, а также поработаем над балансом. Для начала укажем в гиперпараметрах моделей `class_weight = 'balanced'` и посмотрим на показатели. Примечание: совмещать up(down)sampling и class_weight нельзя, так как совмещения двух методов балансирования приводит к новому дисбалансу.

# In[14]:


model = DecisionTreeClassifier()
parametrs = { 'max_depth': range (1, 11),
             'min_samples_leaf': range (1, 11),
             'min_samples_split': range (2, 10) }

# с помощью gridSearch находим лучшие гиперпараметры для модели

grid = GridSearchCV(model, parametrs); 
grid.fit(features_train, target_train);                                         


# Получаем лучшие гиперпараметры для дерева решений.

# In[15]:


grid.best_params_ 


# In[16]:


for depth in range(1, 16, 1):
    model = DecisionTreeClassifier(random_state=12345, max_depth=depth, class_weight = 'balanced')
    model.fit(features_train, target_train)
    predicted_valid = model.predict(features_valid)
    
print("F1:", f1_score(target_valid, predicted_valid))


# In[17]:


#подставляем наилучшие гиперпараметры

model = DecisionTreeClassifier(random_state=12345, max_depth=7, min_samples_leaf = 9, 
                               min_samples_split = 3, class_weight = 'balanced') 

model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

f1_score(target_valid, predicted_valid)   


# Для дерева решений показатель Ф1-меры при подобранных гиперпараметрах и `class_weight = 'balanced`: 0.55

# In[18]:


model = DecisionTreeClassifier(random_state=12345, max_depth=7, min_samples_leaf = 10, min_samples_split = 3)
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

for threshold in np.arange(0, 0.9, 0.02):
    predicted_valid = probabilities_one_valid > threshold
    precision = precision_score(target_valid, predicted_valid)
    recall = recall_score(target_valid, predicted_valid)
    f1score = f1_score(target_valid, predicted_valid) 
    print("Порог = {:.2f} | Точность = {:.3f}, Полнота = {:.3f}".format(
        threshold, precision, recall))
    print(f1score)


# При пороге в 0.32 для дерева решений достигается значение ф1-меры в 0.629

# Далее попробуем случайный лес.

# In[19]:


best_depth = 0
best_est = 0
best_result = 0
for depth in range(1, 16, 1):
    for est in range(10, 60, 10):   
        model = RandomForestClassifier(max_depth=depth, random_state=12345, n_estimators=est, class_weight = 'balanced')
        model.fit(features_train, target_train)
        result = model.score(features_valid, target_valid) 
        if result > best_result:        
            best_result = result   
            best_est = est
            best_depth = depth
        
print(f"Accuracy наилучшей модели на валидационной выборке: {best_result}, Количество деревьев: {best_est}, Максимальная глубина: {best_depth}")


# Сейчас снова обратимся к инструменту GridSearchCV

# In[20]:


model = RandomForestClassifier()
parametrs = { 'max_depth': range (1, 13, 2),
             'n_estimators': range (10, 51, 10),
             'min_samples_leaf': range (1, 11),
             'min_samples_split': range (2, 10, 2) }
grid = GridSearchCV(model, parametrs);
grid.fit(features_train, target_train);


# In[21]:


grid.best_params_


# In[22]:


model = RandomForestClassifier(max_depth=11, random_state=12345, min_samples_leaf = 3, min_samples_split = 4, n_estimators=50, class_weight = 'balanced')
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

f1_score(target_valid, predicted_valid)   


# Для случайного леса показатель Ф1-меры при подобранных гиперпараметрах и `class_weight = 'balanced`: 0.64

# In[23]:


model = RandomForestClassifier(max_depth=11, random_state=12345, min_samples_leaf = 3, min_samples_split = 4, n_estimators=50)
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

for threshold in np.arange(0, 0.8, 0.02):
    predicted_valid = probabilities_one_valid > threshold
    precision = precision_score(target_valid, predicted_valid)
    recall = recall_score(target_valid, predicted_valid)
    f1score = f1_score(target_valid, predicted_valid) 
    print("Порог = {:.2f} | Точность = {:.3f}, Полнота = {:.3f}".format(
        threshold, precision, recall))
    print(f1score)
    
precision, recall, thresholds = precision_recall_curve(target_valid, probabilities_valid[:, 1])

plt.figure(figsize=(6, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Кривая Precision-Recall')
plt.show() 


# При пороге в 0.36 для случайного леса достигается значение ф1-меры в 0.638

# Cудя по всем показателям графиков и метрик, модель случайного леса должна нам подойти и выдать результат Ф1-меры выше 0.59.

# ## Тестирование модели

# Протестируем нашу лучшую модель на тестировочной выборке.

# In[24]:


#первый способ с использованием class_weight = 'balanced'

model = RandomForestClassifier(max_depth=11, random_state=12345, min_samples_leaf = 3, 
                               min_samples_split = 4, n_estimators=50, class_weight = 'balanced')
model.fit(features_train, target_train)
predicted_test = model.predict(features_test)

f1_score(target_test, predicted_test) 


# In[25]:


#второй способ с использованием порога 

#сначала указываем найденные гиперпараметры

model = RandomForestClassifier(random_state=12345, max_depth=11, min_samples_leaf=3, min_samples_split=4, n_estimators= 50)
model.fit(features_train, target_train)
probabilities_test = model.predict_proba(features_test)
probabilities_one_test = probabilities_test[:, 1]

#далее указываем значение порога

threshold = 0.36
predicted_test = probabilities_one_test > threshold
f1score = f1_score(target_test, predicted_test) 
f1score


# In[26]:


fpr, tpr, thresholds = roc_curve(target_test, probabilities_one_test)

plt.figure()
plt.plot(fpr, tpr)

plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.title('ROC-кривая')

plt.show()

auc_roc = roc_auc_score(target_test, probabilities_one_test)
auc_roc


# Нам удалось достичь метрики Ф1-меры в 0.6174 с помощью модели случайного леса.

# Метрика AUC-ROC лучше, чем у случайной модели.

# ## Общий вывод

# **Задачей было построить модель со значением F1-меры более 0.59.**
# 
# Первым этапом, после ознакомления с данными, была проведена их предобработка.
# 1. Были удалены столбцы, которые никак не помогут исследованию и никак не влияют на целевой признак ('customerid', 'rownumber', 'surname')
# 
# 
# 2. Общая выборка была разбита на 3 части в пропорциях 3:1:1  
#  - обучающая выборка
#  - валидационная выборка
#  - тестовая выборка
# 
# Далее было определено, что 20% от общего числа клиентов прекратили пользоваться банком. Дисбаланс классов на лицо в соотношении 20/80.
# 
# 
#     В результате исследования задачи без обработки дизбаланса классов и тщательной настройки моделей, было получено значение Ф1-меры 0.589. Этот результат меньше требуемого, поэтому его требуется улучшать. 
#     
# Мной было принято решение начать борьбу с дисбалансом начиная с использования гиперпараметра модели `class_weight = 'balanced'`, а также найти наилучшие остальные гиперпараметры.
# 
# 
# Вторым способом улучшить баланс и показатели Ф1-меры стало изменение порога классификации.
# 
# 
# **Лучшей моделью оказался случайный лес, который на валидационной выборке выдавал значение Ф1-меры в 0.64.**
# 
# 
# После того, как были подобраны гиперпараметры на валидации  - модель была протестирована на тестовой выборке. 
#  - Значение Ф1-меры достигло 0.6174.
#  - Показатель эффективности модели (AUC-ROC) - 0.853
