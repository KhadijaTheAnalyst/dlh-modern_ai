#!/usr/bin/env python3

import pandas as pd

csv_file = "precleaned-Telco-Customer-Churn.csv"
df = pd.read_csv(csv_file)
print(df.dtypes)
print("\n")
print(df['Churn'].value_counts()['Yes'])
