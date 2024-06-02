import pandas as pd
import os
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

wine_quality = fetch_ucirepo(id=186)

X = wine_quality.data.features
y = wine_quality.data.targets

# Делим данные на тестовую и тренировочную выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Склеиваем X и y для тренировочной и тестовой выборок
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

# Создаём папку для сохранения данных
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)

# Сохраняем данные в файлы
train_data.to_csv('train/train_data.csv', index=False)
test_data.to_csv('test/test_data.csv', index=False)