<img width="1887" height="816" alt="image" src="https://github.com/user-attachments/assets/d66bf62f-40ec-4780-be3f-ee0c2c804e0b" /># ⚙️ Predictive Maintenance — Remaining Useful Life Prediction

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-RMSE%2018.11-orange)
![LSTM](https://img.shields.io/badge/LSTM-RMSE%2014.47-cyan)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![Dataset](https://img.shields.io/badge/Dataset-NASA%20CMAPSS-darkgreen)

A machine learning system that predicts the **Remaining Useful Life (RUL)** of turbofan jet engines using the NASA CMAPSS dataset. Trained XGBoost and LSTM models on multivariate sensor time-series data, with an interactive Streamlit dashboard for real-time RUL prediction and sensor degradation visualization.

---

## 🚀 Live Demo

> [🔗Launch App](https://predictive-maintenance--rul.streamlit.app/)

---

## 📌 What is RUL?

**Remaining Useful Life** = how many operational cycles an engine has left before failure.

In real-world context, one cycle ≈ one flight (engine on → run → off). Airlines use RUL predictions to schedule maintenance *before* failure — avoiding both unplanned breakdowns and unnecessary early servicing.

```
Sensors → Raw Data → ML Model → RUL Prediction → Maintenance Decision
```

---

## 📊 Model Performance (FD001 Subset)

| Model    | RMSE (cycles) |
|----------|--------------|
| XGBoost  | 18.11        |
| LSTM     | **14.47** ✅  |

LSTM outperforms XGBoost because it captures **temporal degradation patterns** in sensor sequences. XGBoost treats each timestep independently and loses sequential context.

---

## 🗂️ Project Structure

```
predictive-maintenance-rul/
├── app.py                  # Streamlit dashboard
├── preprocess.py           # Data loading & feature engineering
├── train_xgboost.py        # XGBoost training script
├── train_lstm.py           # LSTM training script (PyTorch)
├── models/
│   ├── xgb_model.pkl       # Saved XGBoost model
│   ├── scaler.pkl          # XGBoost scaler
│   ├── lstm_model.pt       # Saved LSTM weights
│   └── lstm_scaler.pkl     # LSTM scaler
├── requirements.txt
└── README.md
```

---

## 🧠 Dataset — NASA CMAPSS

**Commercial Modular Aero-Propulsion System Simulation** — a NASA benchmark dataset for predictive maintenance research.

- **Subset used:** FD001 (1 operating condition, HPC degradation fault)
- **Training engines:** 100 units run to failure
- **Test engines:** 100 units (partial run)
- **Sensors:** 21 multivariate time-series readings (temperature, pressure, speed, fuel flow)
- **Low-variance sensors dropped:** s1, s5, s6, s10, s16, s18, s19
- **RUL clipped at 125 cycles** (standard CMAPSS practice — engines don't degrade meaningfully in early cycles)

📥 Download dataset: [Kaggle — NASA CMAPS](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps)

---

## 🏗️ Architecture

### XGBoost
- Features: scaled sensor readings (MinMaxScaler)
- Hyperparameters: 300 estimators, max_depth=6, lr=0.05, subsample=0.8

### LSTM (PyTorch)
- Input: sequences of 30 consecutive cycles per engine
- Architecture: LSTM(128) → Dropout(0.2) → LSTM(64) → Dropout(0.2) → LSTM(32) → Linear(1)
- Optimizer: Adam, Loss: MSE, Early stopping (patience=5)

---

## 📱 Dashboard Features

| Tab | Description |
|-----|-------------|
| 🔮 RUL Prediction | Select engine unit + model → get predicted RUL with delta vs actual |
| 📈 Sensor Analysis | Visualize sensor degradation trends over cycles for any engine unit |
| 📊 Model Comparison | Side-by-side RMSE comparison with bar chart |

Engine health status:
- 🟢 **GOOD** — RUL > 80 cycles
- 🟡 **MODERATE** — RUL 30–80 cycles
- 🔴 **CRITICAL** — RUL < 30 cycles

---

## ⚡ Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/TiKkU12345/predictive-maintenance-rul.git
cd predictive-maintenance-rul
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download dataset**

Download [NASA CMAPS from Kaggle](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps) and place the extracted files at:
```
archive/CMaps/train_FD001.txt
archive/CMaps/test_FD001.txt
archive/CMaps/RUL_FD001.txt
```

**4. Train models**
```bash
python train_xgboost.py
python train_lstm.py
```

**5. Launch dashboard**
```bash
streamlit run app.py
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
xgboost
torch
streamlit
matplotlib
seaborn
```

---

## 🔧 Tech Stack

`Python` · `PyTorch` · `XGBoost` · `Streamlit` · `scikit-learn` · `NASA CMAPSS Dataset`

---

## 👤 Author

**Arunav Kumar (Tikku)**
- GitHub: [@TiKkU12345](https://github.com/TiKkU12345)
- Email: arunav.jsr.0604@gmail.com

---

## 📄 License

MIT License
