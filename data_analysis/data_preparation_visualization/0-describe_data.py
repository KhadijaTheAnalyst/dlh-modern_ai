#!/usr/bin/env python3
"""Print a quick descriptive summary of the Telco churn dataset.

Loads the raw CSV and reports the pieces needed before any cleaning
starts: shape, column dtypes, a preview of the first rows, per-column
missing-value counts, and the number of duplicate rows.
"""
import pandas as pd

df = pd.read_csv('Telco-Customer-Churn.csv')
shape = df.shape
data_types = df.dtypes
head = df.head()
missing_count = df.isnull().sum()
duplicates = df.duplicated().sum()
print("Shape:", shape)
print("Dtypes:\n", data_types)
print("First rows:\n", head)
print("Missing values:\n", missing_count)
print("Duplicates:", duplicates)
