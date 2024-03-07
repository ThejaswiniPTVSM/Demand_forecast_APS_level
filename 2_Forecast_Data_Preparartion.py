# Databricks notebook source
demand_df=spark.sql("""
select distinct
  cast(so.DEALER_ID AS string),
    order_date,
  Part_No,
  cast(round(sum(QUANTITY), 0) as float) as Order_Qty from (select * from nebula.v_dmsoffline_dbo_dp_spare_so_part ) sop left join
(select * from nebula.v_dmsoffline_dbo_dp_dms_spare_sale_order ) so on so.dealer_id = sop.dealer_id
  and so.branch_id = sop.branch_id
  and so.sale_order_id = sop.sale_order_id
  group by 1,2,3
""")

demand_df.write.mode("overwrite").format("delta").partitionBy("dealer_id").saveAsTable("hulk.v_psa_parts_thej_aps_demand")
