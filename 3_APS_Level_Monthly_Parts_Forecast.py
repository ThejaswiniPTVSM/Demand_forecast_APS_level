# Databricks notebook source
# MAGIC %%capture
# MAGIC %pip install autots

# COMMAND ----------

from autots import AutoTS
import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
import matplotlib.pyplot as plt
from pyspark.sql.functions import current_timestamp

# COMMAND ----------

autodealr_id='30267'
fromdate='2021-12-31'
file_path1 = "/Workspace/Users/thejaswini.p@tvsmotor.com/Parts_demand_forecasting/results/auto_ts_forecasts_"+autodealr_id+"_superfast_weekly_M1"
file_path2 = "/Workspace/Users/thejaswini.p@tvsmotor.com/Parts_demand_forecasting/results/auto_ts_forecasts_"+autodealr_id+"_superfast_weekly_M2"
file_path3 = "/Workspace/Users/thejaswini.p@tvsmotor.com/Parts_demand_forecasting/results/auto_ts_forecasts_"+autodealr_id+"_superfast_weekly_M3"

# COMMAND ----------

df = spark.sql(f"""
               SELECT * from hulk.v_psa_parts_thej_aps_demand
               WHERE DEALER_ID = {autodealr_id}
               """).toPandas()
df=df[df['order_date']> fromdate ]
df['Order_Date'] = pd.to_datetime(df['order_date'], format='%Y%m%d')
df=df[['Order_Date','Part_No','Order_Qty']]

# COMMAND ----------

df1=df
def remove_outliers(df, value_column):
    Q1 = df[value_column].quantile(0.25)
    Q3 = df[value_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out the outliers
    df_filtered = df[(df[value_column] >= lower_bound) & (df[value_column] <= upper_bound)]
    return df_filtered
df = df.groupby('Part_No').apply(lambda x: remove_outliers(x, 'Order_Qty')).reset_index(drop=True)

# COMMAND ----------

weekly_data = df.groupby([pd.Grouper(key='Order_Date', freq='W'), 'Part_No']).sum().reset_index()

pivoted_df = weekly_data.pivot(index='Order_Date', columns='Part_No', values='Order_Qty')
pivoted_df.fillna(pivoted_df.mean(), inplace=True)
#pivoted_df.fillna(0, inplace=True)
pivoted_df = pivoted_df.astype(int)
pivoted_df.head(3)

# COMMAND ----------

metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 2,
    'made_weighting': 0.5,
    'mage_weighting': 1,
    'mle_weighting': 0,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'containment_weighting': 0,
    'contour_weighting': 1,
    'runtime_weighting': 0.05,
}

model = AutoTS(
    forecast_length=15, # 
    frequency='W', 
    prediction_interval=0.7,
    ensemble='auto',
    model_list="superfast",  # fast, fast_parallel, scalable, all
    models_to_validate=0.5,
    transformer_list="superfast",   # fast, ...  
    transformer_max_depth=15,
    drop_most_recent=15,
    #drop_data_older_than_periods=60,
    max_generations=15,  # genetic algorithms
    no_negatives=True,
    constraint=3.0,
    prefill_na=0,
    num_validations=10,
    subset=200,
    validation_method="backwards",      # "seasonal 364"
    n_jobs=12,
    verbose=0,
    metric_weighting=metric_weighting
)

model = model.fit(
    pivoted_df)

prediction = model.predict()
df_forecast = prediction.forecast
df_forecast.to_csv(file_path1)

# COMMAND ----------

model = AutoTS(
    forecast_length=10, # 
    frequency='W', 
    prediction_interval=0.7,
    ensemble='auto',
    model_list="superfast",  # fast, fast_parallel, scalable, all
    models_to_validate=0.5,
    transformer_list="superfast",   # fast, ...  
    transformer_max_depth=15,
    drop_most_recent=10,
    #drop_data_older_than_periods=60,
    max_generations=15,  # genetic algorithms
    no_negatives=True,
    constraint=3.0,
    prefill_na=0,
    num_validations=10,
    subset=200,
    validation_method="backwards",      # "seasonal 364"
    n_jobs=12,
    verbose=0,
    metric_weighting=metric_weighting
)

model = model.fit(
    pivoted_df)

prediction = model.predict()
df_forecast = prediction.forecast
df_forecast.to_csv(file_path2)

# COMMAND ----------

metric_weighting = {
    'smape_weighting': 5,
    'mae_weighting': 2,
    'rmse_weighting': 2,
    'made_weighting': 0.5,
    'mage_weighting': 1,
    'mle_weighting': 0,
    'imle_weighting': 0,
    'spl_weighting': 3,
    'containment_weighting': 0,
    'contour_weighting': 1,
    'runtime_weighting': 0.05,
}

model = AutoTS(
    forecast_length=6, # 
    frequency='W',
    prediction_interval=0.7,
    ensemble='auto',
    model_list="superfast",  # fast, fast_parallel, scalable, all
    models_to_validate=0.5,
    transformer_list="superfast",   # fast, ...  
    transformer_max_depth=15,
    drop_most_recent=6,
    #drop_data_older_than_periods=60,
    max_generations=15,  # genetic algorithms
    no_negatives=True,
    constraint=3.0,
    prefill_na=0,
    num_validations=10,
    subset=200,
    validation_method="backwards",      # "seasonal 364"
    n_jobs=12,
    verbose=0,
    metric_weighting=metric_weighting
)

model = model.fit(
    pivoted_df)

prediction = model.predict()
df_forecast = prediction.forecast
df_forecast.to_csv(file_path3)

# COMMAND ----------

abc_fms_df = spark.sql(f"""
               SELECT * from sandbox.abc_fms_tej_4forecast
               WHERE DEALER_ID = {autodealr_id}
               """).toPandas()
abc_fms_df = abc_fms_df.drop_duplicates(subset='PART_NO', keep=False)

# COMMAND ----------

df_forecast_m1 = pd.read_csv(file_path1)
df_forecast_m2 = pd.read_csv(file_path2)
df_forecast_m3 = pd.read_csv(file_path3)

def prepare_results(df):
    df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
    df_long = pd.melt(df, id_vars=['Date'])
    df_long = df_long.rename(columns={'variable': 'PART_NO', 'value': 'PREDICTED_ORDER_QTY'})
    df_long['PREDICTED_ORDER_QTY'] = df_long['PREDICTED_ORDER_QTY'].astype(int)
    return df_long

# Define a function for MAPE
def calculate_mape(actual, forecast):
    return np.abs((actual - forecast) / actual) * 100

# Define a function for SMAPE
def calculate_smape(actual, forecast):
    return 2.0 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast)) * 100

# COMMAND ----------

df1=df.copy()
df1['Order_Date'] = pd.to_datetime(df1['Order_Date'])
#Change the Month Here
df1 = df1[(df1['Order_Date'] > '2023-11-30') & (df1['Order_Date'] < '2024-01-01')]
df1['Week'] = df1['Order_Date'].dt.to_period('W')
actual_weekly = df1.pivot_table(index=['Week','Part_No'], values=['Order_Qty'], aggfunc='sum')
actual_weekly.reset_index(inplace=True)
actual_weekly.rename(columns={'Part_No': 'PART_NO'}, inplace=True)
#print(actual_weekly.head(5))

df_forecast_long_M1 = prepare_results(df_forecast_m1)
df_forecast_long_M1['Date'] = pd.to_datetime(df_forecast_long_M1['Date'])
df_forecast_long_M1 = df_forecast_long_M1[(df_forecast_long_M1['Date'] > '2023-11-30') & (df_forecast_long_M1['Date'] < '2024-01-01')]
df_forecast_long_M1['Week']=df_forecast_long_M1['Date'].dt.to_period('W')
df_forecast_long_M1=df_forecast_long_M1.drop('Date', axis=1)
#print(df_forecast_long_M1.head(5))

result_df_M1 = pd.merge(df_forecast_long_M1,actual_weekly, on=['Week','PART_NO'],how='inner')
result_df_M1=result_df_M1.drop('Week', axis=1)
print(result_week_M1.head(5))
result_df_M1 = result_df_M1.groupby('PART_NO').sum()
result_df_M1['MAPE'] = result_df_M1.apply(lambda row: calculate_mape(row['Order_Qty'], row['PREDICTED_ORDER_QTY']), axis=1)
result_df_M1['SMAPE'] = result_df_M1.apply(lambda row: calculate_smape(row['Order_Qty'], row['PREDICTED_ORDER_QTY']), axis=1)
result_df_M1['month']='2023-12'
result_df_M1.reset_index(inplace=True)
#print(result_df_M1.head(5))

actual_weekly1=actual_weekly[actual_weekly['Order_Qty']>0]
month_wise_actual_parts = actual_weekly1.groupby('Week')['PART_NO'].nunique().reset_index(name='Unique_Parts_Count')
df_forecast_long_M11=df_forecast_long_M1[df_forecast_long_M1['PREDICTED_ORDER_QTY']>0]
month_wise_forecast_parts = df_forecast_long_M11.groupby('Week')['PART_NO'].nunique().reset_index(name='Unique_Parts_Count')
#print(month_wise_actual_parts)
#print(month_wise_forecast_parts)

# COMMAND ----------

df2=df.copy()
df2['Order_Date'] = pd.to_datetime(df2['Order_Date'])
#Change the Month Here
df2 = df2[(df2['Order_Date'] > '2023-12-31') & (df2['Order_Date'] < '2024-02-01')]
df2['Week'] = df2['Order_Date'].dt.to_period('W')
actual_weekly = df2.pivot_table(index=['Week','Part_No'], values=['Order_Qty'], aggfunc='sum')
actual_weekly.reset_index(inplace=True)
actual_weekly.rename(columns={'Part_No': 'PART_NO'}, inplace=True)
#print(actual_weekly.head(5))
#print(actual_weekly['Week'].unique())

df_forecast_long_M2 = prepare_results(df_forecast_m2)
df_forecast_long_M2['Date'] = pd.to_datetime(df_forecast_long_M2['Date'])
df_forecast_long_M2 = df_forecast_long_M2[(df_forecast_long_M2['Date'] > '2023-12-31') & (df_forecast_long_M2['Date'] < '2024-02-01')]
df_forecast_long_M2['Week']=df_forecast_long_M2['Date'].dt.to_period('W')
df_forecast_long_M2=df_forecast_long_M2.drop('Date', axis=1)
#print(df_forecast_long_M2.head(5))
#print(df_forecast_long_M2['Week'].unique())

result_df_M2 = pd.merge(df_forecast_long_M2,actual_weekly, on=['Week','PART_NO'],how='inner')
result_df_M2=result_df_M2.drop('Week', axis=1)
print(result_week_M2.head(5))
result_df_M2 = result_df_M2.groupby('PART_NO').sum()
result_df_M2['MAPE'] = result_df_M2.apply(lambda row: calculate_mape(row['Order_Qty'], row['PREDICTED_ORDER_QTY']), axis=1)
result_df_M2['SMAPE'] = result_df_M2.apply(lambda row: calculate_smape(row['Order_Qty'], row['PREDICTED_ORDER_QTY']), axis=1)
result_df_M2['month']='2024-01'
result_df_M2.reset_index(inplace=True)
#result_df_M2.head(5)

# COMMAND ----------

df3=df.copy()
df3['Order_Date'] = pd.to_datetime(df3['Order_Date'])
#Change the Month Here
df3 = df3[(df3['Order_Date'] >= '2024-02-01') & (df3['Order_Date'] < '2024-03-01')]
df3['Week'] = df3['Order_Date'].dt.to_period('W')
actual_weekly = df3.pivot_table(index=['Week','Part_No'], values=['Order_Qty'], aggfunc='sum')
actual_weekly.reset_index(inplace=True)
actual_weekly.rename(columns={'Part_No': 'PART_NO'}, inplace=True)
#print(actual_weekly.head(5))
#print(actual_weekly['Week'].unique())

df_forecast_long_M3 = prepare_results(df_forecast_m3)
df_forecast_long_M3['Date'] = pd.to_datetime(df_forecast_long_M3['Date'])
df_forecast_long_M3 = df_forecast_long_M3[(df_forecast_long_M3['Date'] >= '2024-02-01') & (df_forecast_long_M3['Date'] < '2024-03-01')]
df_forecast_long_M3['Week']=df_forecast_long_M3['Date'].dt.to_period('W')
df_forecast_long_M3=df_forecast_long_M3.drop('Date', axis=1)
#print(df_forecast_long_M3.head(5))
#print(df_forecast_long_M3['Week'].unique())

result_df_M3 = pd.merge(df_forecast_long_M3,actual_weekly, on=['Week','PART_NO'],how='inner')
result_df_M3=result_df_M3.drop('Week', axis=1)
print(result_week_M3.head(5))
result_df_M3 = result_df_M3.groupby('PART_NO').sum()
result_df_M3['MAPE'] = result_df_M3.apply(lambda row: calculate_mape(row['Order_Qty'], row['PREDICTED_ORDER_QTY']), axis=1)
result_df_M3['SMAPE'] = result_df_M3.apply(lambda row: calculate_smape(row['Order_Qty'], row['PREDICTED_ORDER_QTY']), axis=1)
result_df_M3['month']='2024-02'
result_df_M3.reset_index(inplace=True)
#result_df_M3.head(5)

# COMMAND ----------

result = pd.concat([result_df_M1, result_df_M2, result_df_M3], ignore_index=True)
result_abc_fms=pd.merge(result,abc_fms_df, on=['PART_NO'],how='inner')
result.head(5)

# COMMAND ----------

#result_weekly = result_week_M1
#result_weekly['MAPE'] = result_weekly.apply(lambda row: calculate_mape(row['Order_Qty'], row['PREDICTED_ORDER_QTY']), axis=1)
#result_weekly['SMAPE'] = result_weekly.apply(lambda row: calculate_smape(row['Order_Qty'], row['PREDICTED_ORDER_QTY']), axis=1)
#result_weekly.head(5)

#result_df_M1 = pd.merge(df_forecast_long_M1,actual_weekly, on=['Week','PART_NO'],how='inner')

print(df_forecast_long_M1.head(5))
print(actual_weekly.head(5))
result_df_M1 = pd.merge(actual_weekly,df_forecast_long_M1, on=['Week','PART_NO'],how='inner')
result_df_M1.head(5)
result_df_M1.pivot_table(values=['PREDICTED_ORDER_QTY', 'Order_Qty'], index='Week', aggfunc=np.sum)

# COMMAND ----------

actual_weekly.pivot_table(values=['Order_Qty'], index='Week', aggfunc=np.sum)

# COMMAND ----------

result_abc_fms=pd.merge(result_weekly,abc_fms_df, on=['PART_NO'],how='inner')
result_abc_fms.pivot_table(values=['PREDICTED_ORDER_QTY', 'Order_Qty'], index='Week', aggfunc=np.sum)

# COMMAND ----------

result.pivot_table(values=['MAPE', 'SMAPE'], index='month', aggfunc=np.mean)

# COMMAND ----------

result_abc_fms.pivot_table(values=['MAPE', 'SMAPE'], index='month', columns='Fsn_class',aggfunc=np.mean)

# COMMAND ----------

result_abc_fms.pivot_table(values=['MAPE', 'SMAPE'], index='month', columns='part_type',aggfunc=np.mean)

# COMMAND ----------

result_abc_map=result_abc_fms.pivot_table(values=['MAPE', 'SMAPE'], index='month', columns='ABC_class',aggfunc=np.mean)
result_abc_map.reset_index(inplace=True)
result_abc_map.head(10)

# COMMAND ----------

result_abc_fms=result_abc_fms[result_abc_fms['Order_Qty']>10]
result_abc_fms=result_abc_fms[result_abc_fms['PREDICTED_ORDER_QTY']>10]
result_abc_order=result_abc_fms.pivot_table(values=['Order_Qty', 'PREDICTED_ORDER_QTY'], index='month', columns='ABC_class',aggfunc=np.sum)
result_abc_order.reset_index(inplace=True)
result_abc_order.head(10)

# COMMAND ----------

#for only last 48 weeks and outlier removal
MAPE	SMAPE
CATEGORY	Stranger	repeater	runner	Stranger	repeater	runner
month						
2023-12	32.907093	39.177799	34.735399	41.492985	41.097942	34.303144
2024-01	NaN	        35.963399	48.481657	NaN	37.184762	42.494987
2024-02	91.250000	44.656110	48.297489	55.938697	37.221957	38.258905

# COMMAND ----------

#All data greater than 2021-12-31 weeks and outlier removal
MAPE	SMAPE
CATEGORY	Stranger	repeater	runner	Stranger	repeater	runner
month						
2023-12	15.980125	29.328152	35.029746	24.445057	30.607148	37.599275
2024-01	8.431694	34.110424	44.974863	9.706858	28.581140	35.787595
2024-02	9.412331	36.044115	39.610273	10.620240	29.636098	35.887527

MAPE	SMAPE
ABC_class	A	B	C	A	B	C
month						
2023-12	26.732439	31.399996	31.612329	26.903038	35.366899	36.283520
2024-01	29.274149	35.827312	39.837676	24.311961	29.121270	34.470924
2024-02	30.596899	35.334804	36.153114	24.700443	31.629007	34.230890


# COMMAND ----------

#All data greater than 2022-10-01 weeks and outlier removal
MAPE	SMAPE
CATEGORY	Stranger	repeater	runner	Stranger	repeater	runner
month						
2023-12	15.363935	29.118484	34.549686	21.230545	29.678754	36.969430
2024-01	14.311475	37.776865	51.017958	16.586616	34.447987	43.558452
2024-02	10.439351	36.432053	47.330478	12.634987	31.448803	45.973208
