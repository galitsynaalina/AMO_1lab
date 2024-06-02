import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# Загрузка данных для обучения
train_data = pd.read_csv('train/train_data_scaled.csv', sep=',')

# Разделение предикторов и целевой переменной
X_train = train_data.drop(["quality"], axis=1)
y_train = train_data["quality"]

# Обучение модели
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Сохранение модели
joblib.dump(model, 'trained_model.pkl')