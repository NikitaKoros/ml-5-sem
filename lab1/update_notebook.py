import json

with open('lab1.ipynb', 'r') as f:
    nb = json.load(f)

# Находим ячейку с очисткой выбросов (cell 20) и обновляем
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        code = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
        
        # Обновляем очистку выбросов
        if 'Q1 = train_processed[\'RiskScore\'].quantile(0.25)' in code:
            cell['source'] = '''# Удаляем явные аномалии и применяем IQR с меньшим коэффициентом
train_processed = train_processed[(train_processed['RiskScore'] > -100) & (train_processed['RiskScore'] < 500)]

Q1 = train_processed['RiskScore'].quantile(0.25)
Q3 = train_processed['RiskScore'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 2 * IQR
upper_bound = Q3 + 2 * IQR

print(f"Границы для выбросов: [{lower_bound:.2f}, {upper_bound:.2f}]")
print(f"Количество выбросов: {((train_processed['RiskScore'] < lower_bound) | (train_processed['RiskScore'] > upper_bound)).sum()}")

train_clean = train_processed[
    (train_processed['RiskScore'] >= lower_bound) & 
    (train_processed['RiskScore'] <= upper_bound)
].copy()

print(f"Train shape после очистки выбросов: {train_clean.shape}")
print(f"RiskScore range: [{train_clean['RiskScore'].min():.2f}, {train_clean['RiskScore'].max():.2f}]")'''
        
        # Обновляем feature engineering
        elif 'def create_features(df):' in code and 'AssetToMonthlyIncome' in code:
            cell['source'] = '''def create_features(df):
    df = df.copy()
    
    # Логарифмы для skewed распределений
    for col in ['LoanAmount', 'AnnualIncome', 'MonthlyIncome', 'TotalAssets', 'TotalLiabilities']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col].fillna(0))
    
    # Квадраты важных признаков
    for col in ['CreditScore', 'Age', 'Experience']:
        if col in df.columns:
            df[f'{col}_sq'] = df[col] ** 2
    
    # Соотношения долга
    if 'LoanAmount' in df.columns and 'AnnualIncome' in df.columns:
        df['LoanToIncome'] = df['LoanAmount'] / (df['AnnualIncome'] + 1)
        df['LoanToIncome_sq'] = df['LoanToIncome'] ** 2
    
    if 'MonthlyDebtPayments' in df.columns and 'MonthlyIncome' in df.columns:
        df['MonthlyDebtRatio'] = df['MonthlyDebtPayments'] / (df['MonthlyIncome'] + 1)
        df['MonthlyDebtRatio_sq'] = df['MonthlyDebtRatio'] ** 2
    
    if 'MonthlyLoanPayment' in df.columns and 'MonthlyIncome' in df.columns:
        df['LoanPaymentRatio'] = df['MonthlyLoanPayment'] / (df['MonthlyIncome'] + 1)
    
    # Соотношения активов
    if 'TotalAssets' in df.columns and 'TotalLiabilities' in df.columns:
        df['AssetLiabilityRatio'] = df['TotalAssets'] / (df['TotalLiabilities'] + 1)
        df['NetAssets'] = df['TotalAssets'] - df['TotalLiabilities']
    
    # Взаимодействия
    if 'CreditScore' in df.columns and 'Age' in df.columns:
        df['CreditPerAge'] = df['CreditScore'] / (df['Age'] + 1)
        df['Credit_Age_Product'] = df['CreditScore'] * df['Age'] / 1000
    
    if 'Experience' in df.columns and 'Age' in df.columns:
        df['ExperiencePerAge'] = df['Experience'] / (df['Age'] + 1)
    
    if 'SavingsAccountBalance' in df.columns and 'CheckingAccountBalance' in df.columns:
        df['TotalLiquid'] = df['SavingsAccountBalance'] + df['CheckingAccountBalance']
        df['TotalLiquid_log'] = np.log1p(df['TotalLiquid'])
    
    if 'LoanDuration' in df.columns and 'LoanAmount' in df.columns:
        df['MonthlyLoanBurden'] = df['LoanAmount'] / (df['LoanDuration'] + 1)
    
    if 'MaritalStatus' in df.columns and 'NumberOfDependents' in df.columns:
        df['Family_Status'] = df['MaritalStatus'] * 10 + df['NumberOfDependents']
    
    if 'EmploymentStatus' in df.columns and 'JobTenure' in df.columns:
        df['Employment_Stability'] = df['EmploymentStatus'] * df['JobTenure']
    
    return df

train_clean = create_features(train_clean)
test_processed = create_features(test_processed)

print(f"Создано множество новых признаков включая логарифмы, квадраты и взаимодействия")
print(f"Train shape: {train_clean.shape}")
print(f"Test shape: {test_processed.shape}")'''
        
        # Обновляем удаление высококоррелированных
        elif 'to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.98)]' in code:
            cell['source'] = '''numeric_features = train_clean.select_dtypes(include=[np.number]).columns.tolist()
if 'RiskScore' in numeric_features:
    numeric_features.remove('RiskScore')

corr_matrix = train_clean[numeric_features].corr().abs()

upper_triangle = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.95)]

print(f"Признаки с корреляцией >95%: {to_drop}")
print(f"Количество удаляемых признаков: {len(to_drop)}")

train_clean = train_clean.drop(columns=to_drop)
test_processed = test_processed.drop(columns=[col for col in to_drop if col in test_processed.columns])

print(f"Train shape после удаления: {train_clean.shape}")
print(f"Test shape после удаления: {test_processed.shape}")'''
        
        # Обновляем нормализацию - используем RobustScaler
        elif 'scaler = ZScoreNormalizer()' in code and 'X_train_scaled = scaler.fit_transform' in code:
            cell['source'] = '''from sklearn.preprocessing import RobustScaler

X_train_np = X_train.values
X_test_np = X_test.values

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_np)
X_test_scaled = scaler.transform(X_test_np)

print(f"Данные нормализованы (RobustScaler - устойчив к выбросам)")
print(f"X_train_scaled shape: {X_train_scaled.shape}")
print(f"X_test_scaled shape: {X_test_scaled.shape}")'''
        
        # Обновляем финальную модель - используем Ridge
        elif 'best_params = {' in code and 'method\': \'analytical' in code:
            cell['source'] = '''from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

print("Подбор оптимального alpha для Ridge regression...")
best_alpha = 1.0
best_mse = float('inf')

for alpha in [0.5, 1.0, 5.0, 10.0, 20.0]:
    model = Ridge(alpha=alpha)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    mse = -scores.mean()
    print(f"alpha={alpha:5.1f}: CV MSE = {mse:.4f}")
    if mse < best_mse:
        best_mse = mse
        best_alpha = alpha

print(f"\\nЛучший alpha: {best_alpha}, CV MSE: {best_mse:.4f}")

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

print("✅ Notebook обновлен с улучшенной предобработкой!")
