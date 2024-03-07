# Databricks notebook source
#PTC forcast table
mount_point = "/mnt/sb/rajat_saroj/o0.csv"
 
# Read the CSV file into a PySpark DataFrame
df = spark.read.csv(mount_point , header=True, inferSchema=True)
df.createOrReplaceTempView("forecast_ptc")

# COMMAND ----------

# MAGIC %sql
# MAGIC --APS Meta
# MAGIC select distinct APS_ID,APS_DEALERSHIP_NAME from bumblebee.t_psa_dealer_master_mapping
# MAGIC where APS_ID='30105'

# COMMAND ----------

import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = spark.sql(f"""
               SELECT * from hulk.v_psa_parts_thej_aps_demand
               WHERE DEALER_ID = '30105'
               """).toPandas()
#df=df[df['order_date']> fromdate ]
df['Order_Date'] = pd.to_datetime(df['order_date'], format='%Y%m%d')
df=df[['Order_Date','Order_Qty']]

# COMMAND ----------

#df_part=df[df['Part_No']=='K3010750']
df.set_index('Order_Date', inplace=True)
plt.figure(figsize=(8, 4))
plt.plot(df.index, df['Order_Qty'], color='blue')
plt.title('Parts Demand Over years for 30191')
plt.xlabel('Order_Date')
plt.ylabel('Demand')
plt.grid(True)
plt.show()


# COMMAND ----------

df1=df
df.head(5)

# COMMAND ----------

df=df[df['Order_Date']> '2020-09-01' ]
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df['month_data']=df['Order_Date'].dt.to_period('M')

df.head(5)

# COMMAND ----------

df=df.pivot_table(values=['Order_Qty',], index='month_data',aggfunc=np.sum)
df.reset_index(inplace=True)
df.head(5)

# COMMAND ----------

df1=df1[df1['Order_Date']> '2023-09-01' ]
df1['Order_Date'] = pd.to_datetime(df1['Order_Date'])
df1['Week']=df1['Order_Date'].dt.to_period('W')
df1=df1.pivot_table(values=['Order_Qty',], index='Week',aggfunc=np.sum)
df1.reset_index(inplace=True)

# COMMAND ----------

df1.head(5)

# COMMAND ----------

df=df[df['Order_Date']> '2024-01-20' ]
df['Order_Date'] = pd.to_datetime(df['Order_Date'])
df.head(5)

# COMMAND ----------

df = spark.sql(f"""
               SELECT * from hulk.v_psa_parts_thej_aps_demand
               WHERE DEALER_ID = '30105'
               """).toPandas()
df=df[df['order_date']> '2024-01-31']
df['Order_Date'] = pd.to_datetime(df['order_date'], format='%Y%m%d')
df.head(6)

# COMMAND ----------

import matplotlib.pyplot as plt
df1=df[df['Part_No']=='ND320140']
# Create a box plot
plt.figure(figsize=(6, 4))
plt.boxplot(df1['Order_Qty'], vert=False)  # 'vert=False' for horizontal box plot
plt.title('Box Plot of Order_Qty w.r.t Part_No within Quarter')
plt.xlabel('Order_Qty')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
df1=df[df['Part_No']=='K2070130']
# Create a box plot
plt.figure(figsize=(6, 4))
plt.boxplot(df1['Order_Qty'], vert=False)  # 'vert=False' for horizontal box plot
plt.title('Box Plot of Order_Qty w.r.t Part_No w.r.t Quarter')
plt.xlabel('Order_Qty')
plt.show()

# COMMAND ----------

df1=df
def remove_outliers(df, value_column):
    Q1 = df[value_column].quantile(0.25)
    Q3 = df[value_column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filter out the outliers
    df_filtered = df[(df[value_column] < lower_bound) | (df[value_column] > upper_bound)]
    return df_filtered
df_outlier= df.groupby('Part_No').apply(lambda x: remove_outliers(x, 'Order_Qty')).reset_index(drop=True)


# COMMAND ----------

df_outlier.head(3)

# COMMAND ----------

df1=df[df['Part_No']=='0181013']
# Create a box plot
plt.figure(figsize=(6, 4))
plt.boxplot(df1['Order_Qty'], vert=False)  # 'vert=False' for horizontal box plot
plt.title('Box Plot of Order_Qty w.r.t Part_No w.r.t Week')
plt.xlabel('Order_Qty')
plt.show()

# COMMAND ----------

df1=df[df['Part_No']=='0331318']
# Create a box plot
plt.figure(figsize=(6, 4))
plt.boxplot(df1['Order_Qty'], vert=False)  # 'vert=False' for horizontal box plot
plt.title('Box Plot of Order_Qty w.r.t Part_No w.r.t Week')
plt.xlabel('Order_Qty')
plt.show()

# COMMAND ----------

df_outlier.pivot(index='Part_No', columns='Order_Date', values='Order_Qty')

# COMMAND ----------


