#!/usr/bin/env python3

import os
import pandas as pd
load_mistral = __import__('1-load_mistral').load_mistral
get_response = __import__('2-get_llm_response').get_response
create_dataframe_agent = __import__('7-dataframe_agent').create_dataframe_agent


os.environ["MISTRAL_API_KEY"] = "Insert your key HERE"

llm = load_mistral("mistral-small-latest", 0)

csv_file = "precleaned-Telco-Customer-Churn.csv"
df = pd.read_csv(csv_file)

dataframe_agent = create_dataframe_agent(llm, df)

response_columns = get_response(dataframe_agent,
                                {"input": "List the columns in the dataset and their data types."})
print("Columns and data types:\n", response_columns["output"])
