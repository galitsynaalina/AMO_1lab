import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json

# Загрузка данных для тестирования
test_data = pd.read_csv('test/test_data.csv')

# Разделение предикторов и целевой переменной
X_test = test_data.drop(["quality"], axis=1)
y_test = test_data["quality"]

# Загрузка обученной модели
model = joblib.load('trained_model.pkl')

# Предсказание на тестовых данных
pred = model.predict(X_test)


# Оценка качества модели
mse = mean_absolute_error(y_test, pred)
print(f"Средняя абсолютная ошибка на тестовых данных: {mse}")

with open("testing_mse.json", "w", encoding="utf-8") as json_file:
    json.dump(mse, json_file, ensure_ascii=False)