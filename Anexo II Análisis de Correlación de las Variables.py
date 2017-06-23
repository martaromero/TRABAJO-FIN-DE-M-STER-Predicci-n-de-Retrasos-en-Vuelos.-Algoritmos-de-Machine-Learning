# Databricks notebook source
display(dbutils.fs.ls("dbfs:/dataset/df_preprocesado"))

# COMMAND ----------

import pandas as pd

# Copiamos el archivo para tenerlo en local y poder leerlo con pandas
dbutils.fs.cp("dbfs:/dataset/df_preprocesado/part-00000-e613ac01-a7eb-431c-8718-5c854f2e13ce-c000.csv", "file:/home/ubuntu/databricks/common/df_preprocesado.csv", recurse=False)

df = pd.read_csv("file:/home/ubuntu/databricks/common/df_preprocesado.csv", sep=',', header='infer')

# COMMAND ----------

import seaborn as sns
from matplotlib import pyplot as plt 
import numpy as np
f, ax = plt.subplots(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

# COMMAND ----------

display(f.figure)

# COMMAND ----------

pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', None)

# COMMAND ----------

matrix_correlation=df.corr(method='pearson')
print(matrix_correlation)

# COMMAND ----------

matrix_correlation.to_csv("/home/matrix_correlation.csv", sep=',')

# COMMAND ----------

mc = sqlContext.read.format("csv").option("header", "true").load("file:/home/matrix_correlation.csv")
display(mc)
