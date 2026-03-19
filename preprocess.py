import pandas as pd
import numpy as np

COLUMNS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]

def load_data(base_path):
    train = pd.read_csv(f'{base_path}/train_FD001.txt', sep='\s+', header=None, names=COLUMNS, engine='python')
    test  = pd.read_csv(f'{base_path}/test_FD001.txt',  sep='\s+', header=None, names=COLUMNS, engine='python')
    rul   = pd.read_csv(f'{base_path}/RUL_FD001.txt',   sep='\s+', header=None, names=['RUL'],  engine='python')

    # Drop empty cols
    train.dropna(axis=1, inplace=True)
    test.dropna(axis=1, inplace=True)
    rul.dropna(axis=1, inplace=True)

    # Add RUL to train
    max_cycles = train.groupby('unit')['cycle'].max().reset_index()
    max_cycles.columns = ['unit', 'max_cycle']
    train = train.merge(max_cycles, on='unit')
    train['RUL'] = train['max_cycle'] - train['cycle']
    train.drop('max_cycle', axis=1, inplace=True)

    # Add RUL to test
    test_last = test.groupby('unit').last().reset_index()
    
    # Verify unit count matches
    print(f'Test units: {len(test_last)}, RUL entries: {len(rul)}')
    assert len(test_last) == len(rul), "Mismatch between test units and RUL entries!"
    
    test_last['RUL'] = rul['RUL'].values

    return train, test, test_last

def get_features():
    drop_cols = ['unit', 'cycle', 'op1', 'op2', 'op3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
    return drop_cols

def prepare_xy(df, drop_cols):
    X = df.drop(columns=drop_cols + ['RUL'], errors='ignore')
    y = df['RUL'] if 'RUL' in df.columns else None
    return X, y