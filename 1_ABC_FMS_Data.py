# Databricks notebook source
import pandas as pd
from pyspark.sql.functions import when, col, concat, lit, date_format, to_date, asc, rand, concat_ws, monotonically_increasing_id, expr

# COMMAND ----------

# Import necessary libraries
import numpy as np
from pyspark.sql.functions import concat, col

# Primary sale query
# Query to extract sales data for parts within a specified date range and order types
df = spark.sql(f""" 
    SELECT * FROM (
        SELECT 
            dealer_id, 
            Part_no,
            COUNT(Part_no) AS num_of_order_placed,
            SUM(Total_billing_amount) AS value,
            SUM(Order_qty) AS qty 
        FROM bumblebee.t_psa_parts_sales_tvs 
        WHERE  
            (Order_date BETWEEN '2021-12-31' AND '2024-03-01') 
            AND order_type IN ('ZCCP','ZSIF','ZSTK','ZVOR','ZHST','ZBLK','ZRED','ZYEW','ZGRN') 
            AND dealer_id IN (SELECT DISTINCT(DEALER_ID) FROM bumblebee.t_psa_dealer_master_mapping)
        GROUP BY 1,2
    ) 
""").toPandas()

# ABC Analysis
# Grouping sales data by dealer and part number and calculating total value of sales per dealer
ABC_data= df.groupby(['dealer_id', 'Part_no'])['value'].sum().reset_index()

# Calculating total value of sales per dealer
ABC_data_country= df.groupby(['dealer_id'])['value'].sum().to_frame(name = "dealer_id_sum_value").reset_index()

# Merging total sales data with individual part sales data
ABC_data=pd.merge(ABC_data, ABC_data_country, on =['dealer_id'])

# Calculating percentage contribution of each part's sales to the total sales of its dealer
ABC_data['pct_value']= ABC_data['value'].astype(float)/ABC_data['dealer_id_sum_value'].astype(float)*100

# Sorting data by dealer ID and percentage value in descending order
ABC_data=ABC_data.sort_values(by = ['dealer_id','pct_value'], ascending=False)

# Converting data types
ABC_data['pct_value'] = ABC_data['pct_value'].astype('float')
ABC_data['value'] = ABC_data['value'].astype('float')

# Calculating cumulative sum of percentage values
ABC_data['cum_sum_value']= ABC_data.groupby(['dealer_id'])['pct_value'].cumsum(axis=0)

# Categorizing parts into A1, A2, A3 based on their contribution to total sales
ABC_data['ABC_category_new'] =  pd.qcut(ABC_data['value'], q=[0, .8 , .95, 1], labels=['A', 'B', 'C'], precision=0)

# FMS Analysis
# Grouping sales data by dealer and part number and calculating total number of orders placed for each part
FMS_data_1= df.groupby(['dealer_id', 'Part_no'])['num_of_order_placed'].sum().reset_index()

# Calculating total number of orders placed per dealer
FMS_data_country= df.groupby(['dealer_id'])['num_of_order_placed'].sum().to_frame(name = "dealer_id_sum_qty").reset_index()

# Merging total order data with individual part order data
FMS_data_1=pd.merge(FMS_data_1, FMS_data_country, on =['dealer_id'])

# Calculating percentage contribution of each part's orders to the total orders of its dealer
FMS_data_1['pct_qty']= FMS_data_1['num_of_order_placed']/FMS_data_1['dealer_id_sum_qty']*100

# Sorting data by dealer ID and percentage quantity in descending order
FMS_data_1= FMS_data_1.sort_values(by = ['dealer_id','pct_qty'], ascending=False)

# Converting data types
FMS_data_1['pct_qty'] = FMS_data_1['pct_qty'].astype('float')
FMS_data_1['num_of_order_placed'] = FMS_data_1['num_of_order_placed'].astype('float')

# Calculating cumulative sum of percentage quantities
FMS_data_1['cum_sum_qty']= FMS_data_1.groupby(['dealer_id'])['pct_qty'].cumsum(axis=0)

# Categorizing parts into C1, C2, C3 based on their contribution to total orders
FMS_data_1['FMS_category'] =np.where(FMS_data_1['cum_sum_qty']<= 80, "F", np.where(FMS_data_1['cum_sum_qty']<=95, "M", "S" ))                            

# Merging ABC and FMS data
ABC_FMS_data_final=pd.merge(ABC_data, FMS_data_1,on =['dealer_id', 'Part_no'])

# Creating a Spark DataFrame from the final merged data
ABC_FMS_data_final1 = spark.createDataFrame(ABC_FMS_data_final)

# Concatenating category columns into a single column
ABC_FMS_data_final1 = ABC_FMS_data_final1.withColumn("merged_column", concat(col("ABC_category_new"), col("FMS_category")))

# Creating temporary views for further analysis
ABC_FMS_data_final1.createOrReplaceTempView("test")
ABC_FMS_data_final1.createOrReplaceTempView("primary")


# COMMAND ----------

abc_fms_sdf = spark.sql("""select dealer_id as DEALER_ID,ABC_category_new as ABC_class,FMS_category as Fsn_class,part_no as PART_NO,
 case
        when concat(coalesce(ABC_category_new, ABC_category_new), coalesce(FMS_category, FMS_category)) in ('AF', 'AM', 'BF') then 'Runner'
        when concat(coalesce(ABC_category_new, ABC_category_new), coalesce(FMS_category, FMS_category)) in ('BM', 'CF') then 'Repeater'
        else 'Stranger'
            end as part_type
        from primary;""")
        #where  DEALER_ID = '30191'
abc_fms_sdf.write.mode('overwrite').saveAsTable('sandbox.abc_fms_tej_4forecast')
