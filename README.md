import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
data = pd.read_csv('C:/Users/Lenovo/Documents/salarykz.csv')
data['Score'] = data['Score'].str.replace(',', '').astype(float)
X = data[['Rank', 'SubmissionCount']]
y = data['Score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test.round(), predictions.round())
print(f'Точность модели: {accuracy}')
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, color='blue', label='Actual vs. Predicted')
plt.xlabel('Реальные значения')
plt.ylabel('Предсказанные значения')
plt.title('График реальных и предсказанных значений')
plt.legend()
plt.show()
