#!/usr/bin/env python3

import pandas as pd
plot_missingness = __import__('1-plot_missingness').plot_missingness


df = pd.read_csv(r"C:\dlh-modern_ai\data_analysis\data_preparation_visualization\Telco-Customer-Churn.csv")
plot_missingness(df)