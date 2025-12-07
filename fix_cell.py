import json

with open('lab1.ipynb', 'r') as f:
    nb = json.load(f)

# Находим и исправляем ячейку с финальной моделью
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        code = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        if 'Подбор оптимального alpha для Ridge regression' in code:
            cell['source'] = '''from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

print("Подбор оптимального alpha для Ridge regression...")
best_alpha = 1.0
best_cv_mse = float('inf')

for alpha in [0.5, 1.0, 5.0, 10.0, 20.0]:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    cv_mse = -scores.mean()
    print(f"alpha={alpha:5.1f}: CV MSE = {cv_mse:.4f}")
    if cv_mse < best_cv_mse:
        best_cv_mse = cv_mse
        best_alpha = alpha

print(f"\\nЛучший alpha: {best_alpha}, CV MSE: {best_cv_mse:.4f}")

final_model = Ridge(alpha=best_alpha)
final_model.fit(X_train_scaled, y_train)

y_train_pred = final_model.predict(X_train_scaled)

print("\\nМетрики финальной модели на обучающей выборке:")
print(f"MSE: {mse(y_train, y_train_pred):.4f}")
print(f"MAE: {mae(y_train, y_train_pred):.4f}")
print(f"R2: {r2(y_train, y_train_pred):.4f}")
print(f"MAPE: {mape(y_train, y_train_pred):.4f}%")'''

with open('lab1.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("✅ Исправлено!")
