import pandas as pd
import numpy as np


def Precision(df, t):
    TP = df[(df['prob'] >= t) & (df['label'] == 'Спелый')].shape[0]
    FP = df[(df['prob'] >= t) & (df['label'] == 'Неспелый')].shape[0]
    if TP == FP == 0:
        return 0
    else:
        return TP / (TP + FP)

df = pd.DataFrame({
    'prob': [0.85, 0.55, 0.65, 0.40, 0.95, 0.75, 0.50, 0.60, 0.30, 0.80],
    'label': ['Спелый', 'Спелый', 'Неспелый', 'Спелый', 'Спелый', 'Неспелый', 'Спелый', 'Спелый', 'Неспелый', 'Спелый']
})

n = 1000
AP = 0

for t in np.linspace(0, 1, n):
    AP += (1 / n) * Precision(df, t)

print(f"{AP:.2}")