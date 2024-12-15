#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
from pyspark.sql import SparkSession

# Инициализация SparkSession
spark = SparkSession.builder \
    .appName("HDFS_Spark_Hive_Analytics") \
    .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
    .enableHiveSupport() \
    .getOrCreate()

# Проверка подключения к HDFS
os.system("hdfs dfs -ls /")


# In[ ]:


df = spark.read.csv("hdfs://namenode:9000/data/input_data.csv", header=True, inferSchema=True)
# Показать данные
df.show()
# Применение трансформаций (например, группировка и агрегация)
result_df = df.groupBy("category").agg({"value": "sum"})
# Запись результата обратно в HDFS
result_df.write.csv("hdfs://namenode:9000/data/output_data.csv", header=True)


# In[ ]:


spark.sql("""
CREATE TABLE IF NOT EXISTS analytics_data (
    category STRING,
    value FLOAT
) USING hive
""")
# Заполнение таблицы данными из DataFrame
result_df.createOrReplaceTempView("temp_data")
spark.sql("""
INSERT INTO TABLE analytics_data
SELECT * FROM temp_data
""")
# Выполнение SQL-запроса к таблице Hive
hive_results = spark.sql("SELECT * FROM analytics_data WHERE value > 100")
hive_results.show()


# In[ ]:


import matplotlib.pyplot as plt
# Преобразуем результаты в Pandas DataFrame для визуализации
hive_results_pd = hive_results.toPandas()
# Визуализация данных
hive_results_pd.plot(kind='bar', x='category', y='value', title='Analytics Data', legend=False)
plt.ylabel('Sum of Values')
plt.show()


# In[ ]:




