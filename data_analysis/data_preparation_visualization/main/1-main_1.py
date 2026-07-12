#!/usr/bin/env python3

import numpy as np
import pandas as pd
plot_missingness = __import__('1-plot_missingness').plot_missingness


data = {
    'customerID': ['A', 'B', 'C', 'A', 'D', 'E'],
    'gender': ['Male', 'Female', np.nan, 'Male', 'Female', 'Male'],
    'Partner': ['Yes', 'No', 'No', np.nan, 'Yes', None],
    'tenure': [1, 24, 5, 1, 36, np.nan],
    'TotalCharges': [29.85, 2138.40, np.nan, 29.85, 3803.40, 0],
    'Churn': ['No', 'Yes', np.nan, 'No', None, 'No']
}
test_df = pd.DataFrame(data)
print(test_df)
plot_missingness(test_df)