#!/usr/bin/env python3

import pandas as pd


df = pd.read_csv(r"C:\dlh-modern_ai\data_analysis\data_preparation_visualization\Telco-Customer-Churn.csv")
non_numeric = [
    val for val in df['TotalCharges'].unique()
    if not str(val).replace('.', '').replace('-', '').isdigit()
]
print("Non-numeric values: ", non_numeric)
