import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle
import os
from preprocess import load_data, get_features, prepare_xy

BASE_PATH = r'D:\D drive\Ml Sys\archive\CMaps'
MODEL_DIR = r'D:\D drive\Ml Sys\models'
os.makedirs(MODEL_DIR, exist_ok=True)

# Load data
train, test, test_last = load_data(BASE_PATH)
drop_cols = get_features()

# Clip RUL at 125 (standard CMAPSS practice - engine doesn't degrade much early on)
train['RUL'] = train['RUL'].clip(upper=125)

# Prepare
X_train, y_train = prepare_xy(train, drop_cols)
X_test, y_test   = prepare_xy(test_last, drop_cols)

# Scale
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train
model = xgb.XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# Evaluate
preds = model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f'XGBoost RMSE: {rmse:.2f} cycles')

# Save model + scaler
pickle.dump(model, open(f'{MODEL_DIR}/xgb_model.pkl', 'wb'))
pickle.dump(scaler, open(f'{MODEL_DIR}/scaler.pkl', 'wb'))
print('Model and scaler saved.')