import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
import os 
matplotlib.use('Agg')
from preprocess import load_data, get_features

# ── Config
# Naya (relative path) — LAGAO
BASE_PATH  = os.path.join(os.path.dirname(__file__), 'archive', 'CMaps')
MODEL_DIR  = os.path.join(os.path.dirname(__file__), 'models')
SEQ_LEN    = 30

st.set_page_config(page_title='Predictive Maintenance', page_icon='⚙️', layout='wide')

# ── LSTM Model Class (must match train_lstm.py) 
class LSTMModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, 128, batch_first=True)
        self.drop1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.drop2 = nn.Dropout(0.2)
        self.lstm3 = nn.LSTM(64, 32, batch_first=True)
        self.fc    = nn.Linear(32, 1)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x = self.drop1(x)
        x, _ = self.lstm2(x)
        x = self.drop2(x)
        x, _ = self.lstm3(x)
        return self.fc(x[:, -1, :]).squeeze()

# ── Load Data & Models (cached) 
@st.cache_data
def load_all_data():
    train, test, test_last = load_data(BASE_PATH)
    return train, test, test_last

@st.cache_resource
def load_models():
    # XGBoost
    xgb_model  = pickle.load(open(f'{MODEL_DIR}/xgb_model.pkl', 'rb'))
    xgb_scaler = pickle.load(open(f'{MODEL_DIR}/scaler.pkl',    'rb'))

    # LSTM
    lstm_scaler = pickle.load(open(f'{MODEL_DIR}/lstm_scaler.pkl', 'rb'))
    # Get input size from scaler
    input_size = lstm_scaler.n_features_in_
    lstm_model = LSTMModel(input_size)
    lstm_model.load_state_dict(torch.load(f'{MODEL_DIR}/lstm_model.pt', map_location='cpu'))
    lstm_model.eval()

    return xgb_model, xgb_scaler, lstm_model, lstm_scaler

train, test, test_last = load_all_data()
xgb_model, xgb_scaler, lstm_model, lstm_scaler = load_models()
drop_cols    = get_features()
feature_cols = [c for c in train.columns if c not in drop_cols + ['RUL']]

# ── Helper: LSTM prediction for one unit 
def predict_lstm_unit(unit_id):
    unit_df  = test[test['unit'] == unit_id].reset_index(drop=True)
    features = lstm_scaler.transform(unit_df[feature_cols].values).astype(np.float32)
    if len(features) >= SEQ_LEN:
        seq = features[-SEQ_LEN:]
    else:
        pad = np.zeros((SEQ_LEN - len(features), features.shape[1]), dtype=np.float32)
        seq = np.vstack([pad, features])
    tensor = torch.tensor(seq).unsqueeze(0)
    with torch.no_grad():
        pred = lstm_model(tensor).item()
    return max(0, round(pred))

# ── Helper: XGBoost prediction for one unit 
def predict_xgb_unit(unit_id):
    row     = test_last[test_last['unit'] == unit_id][feature_cols]
    scaled  = xgb_scaler.transform(row)
    pred    = xgb_model.predict(scaled)[0]
    return max(0, round(pred))

# ── Sidebar 
st.sidebar.image('Nasa.png', width=80)
st.sidebar.title('⚙️ Predictive Maintenance')
st.sidebar.markdown('NASA CMAPSS Turbofan Engine Dataset')
st.sidebar.divider()
tab_choice = st.sidebar.radio('Navigate', ['🔮 RUL Prediction', '📈 Sensor Analysis', '📊 Model Comparison'])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RUL Prediction
# ══════════════════════════════════════════════════════════════════════════════
if tab_choice == '🔮 RUL Prediction':
    st.title('🔮 Remaining Useful Life Prediction')
    st.markdown('Select an engine unit and model to predict how many cycles remain before failure.')
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        unit_id = st.selectbox('Select Engine Unit', sorted(test['unit'].unique()))
    with col2:
        model_choice = st.selectbox('Select Model', ['XGBoost', 'LSTM', 'Both'])

    if st.button('🚀 Predict RUL', use_container_width=True):
        actual_rul = int(test_last[test_last['unit'] == unit_id]['RUL'].values[0])

        if model_choice == 'XGBoost':
            pred = predict_xgb_unit(unit_id)
            st.metric('XGBoost Predicted RUL', f'{pred} cycles', delta=f'{pred - actual_rul} vs actual')
            st.info(f'Actual RUL: **{actual_rul} cycles**')

        elif model_choice == 'LSTM':
            pred = predict_lstm_unit(unit_id)
            st.metric('LSTM Predicted RUL', f'{pred} cycles', delta=f'{pred - actual_rul} vs actual')
            st.info(f'Actual RUL: **{actual_rul} cycles**')

        else:  # Both
            xgb_pred  = predict_xgb_unit(unit_id)
            lstm_pred = predict_lstm_unit(unit_id)

            c1, c2, c3 = st.columns(3)
            c1.metric('Actual RUL',          f'{actual_rul} cycles')
            c2.metric('XGBoost Prediction',  f'{xgb_pred} cycles',  delta=f'{xgb_pred  - actual_rul}')
            c3.metric('LSTM Prediction',     f'{lstm_pred} cycles', delta=f'{lstm_pred - actual_rul}')

        # Health status
        st.divider()
        health = actual_rul
        if health > 80:
            st.success('🟢 Engine Health: GOOD — No immediate maintenance required')
        elif health > 30:
            st.warning('🟡 Engine Health: MODERATE — Schedule maintenance soon')
        else:
            st.error('🔴 Engine Health: CRITICAL — Immediate maintenance required')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Sensor Analysis
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == '📈 Sensor Analysis':
    st.title('📈 Sensor Degradation Analysis')
    st.markdown('Visualize how sensor readings change over the engine lifecycle.')
    st.divider()

    unit_id = st.selectbox('Select Engine Unit', sorted(train['unit'].unique()))
    sensors = [c for c in feature_cols if c.startswith('s')]
    selected_sensors = st.multiselect('Select Sensors to Plot', sensors, default=sensors[:4])

    if selected_sensors:
        unit_df = train[train['unit'] == unit_id].reset_index(drop=True)

        fig, axes = plt.subplots(len(selected_sensors), 1,
                                  figsize=(12, 3 * len(selected_sensors)),
                                  facecolor='#0e1117')
        if len(selected_sensors) == 1:
            axes = [axes]

        for ax, sensor in zip(axes, selected_sensors):
            ax.set_facecolor('#1a1f2e')
            ax.plot(unit_df['cycle'], unit_df[sensor], color='#00d4ff', linewidth=1.5)
            ax.set_ylabel(sensor, color='white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
            ax.grid(True, alpha=0.2)

        axes[-1].set_xlabel('Cycle', color='white')
        fig.suptitle(f'Sensor Readings — Engine Unit {unit_id}', color='white', fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    else:
        st.warning('Please select at least one sensor.')

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ══════════════════════════════════════════════════════════════════════════════
elif tab_choice == '📊 Model Comparison':
    st.title('📊 Model Performance Comparison')
    st.divider()

    col1, col2 = st.columns(2)
    col1.metric('XGBoost RMSE', '18.11 cycles')
    col2.metric('LSTM RMSE',    '14.47 cycles', delta='-3.64 better', delta_color='inverse')

    # Bar chart
    fig, ax = plt.subplots(figsize=(7, 4), facecolor='#0e1117')
    ax.set_facecolor('#1a1f2e')
    models = ['XGBoost', 'LSTM']
    rmses  = [18.11, 14.47]
    bars   = ax.bar(models, rmses, color=['#f97316', '#00d4ff'], width=0.4)
    ax.set_ylabel('RMSE (cycles)', color='white')
    ax.set_title('RMSE Comparison on FD001 Test Set', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{val}', ha='center', color='white', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.markdown('''
    **Why LSTM outperforms XGBoost here:**
    - LSTM captures **temporal dependencies** in sensor degradation sequences
    - XGBoost treats each timestep independently — loses sequential context
    - For time-series degradation data, sequence models have a natural edge
    ''')
