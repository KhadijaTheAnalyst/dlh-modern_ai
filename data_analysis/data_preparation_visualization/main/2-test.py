#!/usr/bin/env python3

import pandas as pd
convert_columns = __import__('2-convert_columns').convert_columns

df = pd.read_csv('Telco-Customer-Churn.csv')
df = convert_columns(df)
pd.set_option('display.max_columns', 12)
print(df[df['TotalCharges'].isna()])