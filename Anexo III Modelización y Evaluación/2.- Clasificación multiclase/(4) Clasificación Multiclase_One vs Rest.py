# Databricks notebook source
# SOLO PERMITE LA CLASIFICACION MULTICLASE

# COMMAND ----------

# (0: retrasos de entre 0-14 min, 1: retrasos entre 15-29 min, 2: retrasos entre 30-44 min, 3: retrasos entre 45-59 min, 4: retrasos entre 60-74 min, 
# 5: retrasos entre 75-89 min, 6: retrasos entre 90-104 min, 7: retrasos entre 105-119 min, 8: retrasos entre 120-134 min, 9: retrasos entre 135-149 min, 10: retrasos entre 150-164 min, 11: retrasos entre 165-179 min, 12: retrasos de mas de 180 min, 13: llegadas entre 15-1 min de antelacion, 14: llegadas de mas de 15 min de antelacion)

# COMMAND ----------

#PARTE1: DEFINIMOS LOS TIPOS DE DATOS DEL ARCHIVO TRAIN 
#Leemos el archivo csv con las cabeceras, importandolo primero todo como string

df = sqlContext.read.format("csv").option("header", "true").load("dbfs:/dataset/datos_preprocesados.csv", inferSchema='true')

from pyspark.sql.types import *
from pyspark.sql.functions import trim, col

df=df.select(trim(col("MONTH")).cast(IntegerType()).alias("MONTH"),
                trim(col("HOLIDAYS")).cast(IntegerType()).alias("HOLIDAYS"),
                trim(col("DAY_OF_MONTH")).cast(IntegerType()).alias("DAY_OF_MONTH"),
                trim(col("DAY_OF_WEEK")).cast(IntegerType()).alias("DAY_OF_WEEK"),
                trim(col("UNIQUE_CARRIER")).cast(IntegerType()).alias("UNIQUE_CARRIER"),
                trim(col("TAIL_NUM")).cast(IntegerType()).alias("TAIL_NUM"),
                trim(col("FL_NUM")).cast(IntegerType()).alias("FL_NUM"),
                trim(col("ORIGIN_AIRPORT_ID")).cast(IntegerType()).alias("ORIGIN_AIRPORT_ID"),
                trim(col("ORIGIN_CITY_MARKET_ID")).cast(IntegerType()).alias("ORIGIN_CITY_MARKET_ID"),
                trim(col("ORIGIN_STATE_NM")).cast(IntegerType()).alias("ORIGIN_STATE_NM"),
                trim(col("DEST_AIRPORT_ID")).cast(IntegerType()).alias("DEST_AIRPORT_ID"),
                trim(col("DEST_CITY_MARKET_ID")).cast(IntegerType()).alias("DEST_CITY_MARKET_ID"),
                trim(col("DEST_STATE_NM")).cast(IntegerType()).alias("DEST_STATE_NM"),
                trim(col("CRS_DEP_TIME")).cast(IntegerType()).alias("CRS_DEP_TIME"),
                trim(col("DEP_TIME")).cast(IntegerType()).alias("DEP_TIME"),
                trim(col("DEP_DELAY")).cast(IntegerType()).alias("DEP_DELAY"),
                trim(col("DEP_DELAY_NEW")).cast(IntegerType()).alias("DEP_DELAY_NEW"),
                trim(col("DEP_DEL15")).cast(IntegerType()).alias("DEP_DEL15"),
                trim(col("DEP_DELAY_GROUP")).cast(IntegerType()).alias("DEP_DELAY_GROUP"),
                trim(col("DEP_TIME_BLK")).cast(IntegerType()).alias("DEP_TIME_BLK"),
                trim(col("TAXI_OUT")).cast(IntegerType()).alias("TAXI_OUT"),
                trim(col("WHEELS_OFF")).cast(IntegerType()).alias("WHEELS_OFF"),
#                 trim(col("TAXI_IN")).cast(IntegerType()).alias("TAXI_IN"),
                trim(col("CRS_ARR_TIME")).cast(IntegerType()).alias("CRS_ARR_TIME"),
#                 trim(col("ARR_TIME")).cast(IntegerType()).alias("ARR_TIME"),
#                 trim(col("ARR_DELAY")).cast(IntegerType()).alias("ARR_DELAY"),
#                 trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
#                 trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"), 
                trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("label"), #COLUMNA A PREDECIR
                trim(col("ARR_TIME_BLK")).cast(IntegerType()).alias("ARR_TIME_BLK"),
                trim(col("CRS_ELAPSED_TIME_numero")).cast(IntegerType()).alias("CRS_ELAPSED_TIME_numero"), 
#                 trim(col("ACTUAL_ELAPSED_TIME_numero")).cast(IntegerType()).alias("ACTUAL_ELAPSED_TIME_numero"),
                trim(col("DISTANCE_GROUP")).cast(IntegerType()).alias("DISTANCE_GROUP"),
                trim(col("WEATHER_DELAY")).cast(IntegerType()).alias("WEATHER_DELAY"),
                trim(col("N_FLIGHTS_DAY_ORIGIN")).cast(IntegerType()).alias("N_FLIGHTS_DAY_ORIGIN"),
                trim(col("N_FLIGHTS_DAY_DEST")).cast(IntegerType()).alias("N_FLIGHTS_DAY_DEST"),
                trim(col("N_FLIGHTS_HOUR_ORIGIN")).cast(IntegerType()).alias("N_FLIGHTS_HOUR_ORIGIN"),
                trim(col("N_FLIGHTS_HOUR_DEST")).cast(IntegerType()).alias("N_FLIGHTS_HOUR_DEST"),
                trim(col("ORDER_FLIGHT_ORIGIN")).cast(IntegerType()).alias("ORDER_FLIGHT_ORIGIN"),
                trim(col("DEP_DELAY_CUM")).cast(IntegerType()).alias("DEP_DELAY_CUM"),
                trim(col("ORDER_FLIGHT_DEST")).cast(IntegerType()).alias("ORDER_FLIGHT_DEST"),
                trim(col("ARR_DELAY_CUM")).cast(IntegerType()).alias("ARR_DELAY_CUM"),
                trim(col("ARR_DELAY_FNUMBER_BEFORE")).cast(IntegerType()).alias("ARR_DELAY_FNUMBER_BEFORE"),
                trim(col("CUM_ARR_FNUMBER_DELAY")).cast(IntegerType()).alias("CUM_ARR_FNUMBER_DELAY"),
                trim(col("DEP_DELAY_FNUMBER_BEFORE")).cast(IntegerType()).alias("DEP_DELAY_FNUMBER_BEFORE"),
                trim(col("CUM_DEP_FNUMBER_DELAY")).cast(IntegerType()).alias("CUM_DEP_FNUMBER_DELAY")                        
               )

# COMMAND ----------

# Definimos nuestros datos de entrenamiento del modelo ("train") y los de prediccion ("test")
train=df.where(df.MONTH!=12)
test=df.where(df.MONTH==12)

#Dividimos el archivo en train a su vez en: train(90%) y evaluation(10%)
(train, evaluation) = train.randomSplit((0.9, 0.1))

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

#Generamos un vector con la columna label y la columna array features
ignore = ['label']
assembler = VectorAssembler(inputCols=[x for x in train.columns if x not in ignore], outputCol='features')
train_LP = assembler.transform(train).select(['label', 'features']) 
evaluation_LP = assembler.transform(evaluation).select(['label', 'features'])

#Definimos el algoritmo del modelo (OvR)
# instantiate the base classifier. Only LogisticRegression and NaiveBayes are supported
lr = LogisticRegression(maxIter=20, tol=1E-6, fitIntercept=True) # elasticNetParam=0.1
# instantiate the One Vs Rest Classifier. 
ovr = OneVsRest(classifier=lr)

# Fit the model
# train the multiclass model.
ovrModel = ovr.fit(train_LP)

# Make predictions.
# score the model on test data.
predictions = ovrModel.transform(evaluation_LP)

# Select (prediction, true label) and compute evaluation error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" %(1.0 - accuracy))

# maxIter=10, tol=1E-6, fitIntercept=True
# Error = 0.519118

# maxIter=50, regParam=0.05, elasticNetParam=0.05
# Error = 0.531474

# maxIter=20, tol=1E-6, fitIntercept=True
# Error = 0.511808

# COMMAND ----------

# Predecimos los grupos de retraso en el mes de diciembre (test)

from pyspark.ml.feature import VectorAssembler

ignore=["label"]
assembler = VectorAssembler(inputCols=[x for x in test.columns if x not in ignore], outputCol='features')
test_LP = assembler.transform(test).select(["label",'features'])

# COMMAND ----------

prediccion_december=ovrModel.transform(test_LP).select(['label','prediction'])
display(prediccion_december)

# COMMAND ----------

pd_december=prediccion_december.toPandas()

# COMMAND ----------

import pandas as pd
pd.crosstab(pd_december["label"], pd_december["prediction"], rownames=['True'], colnames=['Predicted'], margins=True)

# COMMAND ----------

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pd_december["label"], pd_december["prediction"])
print("Accuracy = %g" %(accuracy*100))

from sklearn.metrics import classification_report
report = classification_report(pd_december["label"], pd_december["prediction"], digits=4)
print(report)

from sklearn.metrics import cohen_kappa_score
cohen_kappa_score = cohen_kappa_score(pd_december["label"], pd_december["prediction"])
print("Cohens Kappa = %g" %cohen_kappa_score)

# COMMAND ----------

# NORMALIZANDO DATOS

# COMMAND ----------

#PARTE1: DEFINIMOS LOS TIPOS DE DATOS DEL ARCHIVO TRAIN 
#Leemos el archivo csv con las cabeceras, importandolo primero todo como string

df = sqlContext.read.format("csv").option("header", "true").load("dbfs:/dataset/datos_preprocesados.csv", inferSchema='true')

from pyspark.sql.types import *
from pyspark.sql.functions import trim, col

df=df.select(trim(col("MONTH")).cast(IntegerType()).alias("MONTH"),
                trim(col("HOLIDAYS")).cast(IntegerType()).alias("HOLIDAYS"),
                trim(col("DAY_OF_MONTH")).cast(IntegerType()).alias("DAY_OF_MONTH"),
                trim(col("DAY_OF_WEEK")).cast(IntegerType()).alias("DAY_OF_WEEK"),
                trim(col("UNIQUE_CARRIER")).cast(IntegerType()).alias("UNIQUE_CARRIER"),
                trim(col("TAIL_NUM")).cast(IntegerType()).alias("TAIL_NUM"),
                trim(col("FL_NUM")).cast(IntegerType()).alias("FL_NUM"),
                trim(col("ORIGIN_AIRPORT_ID")).cast(IntegerType()).alias("ORIGIN_AIRPORT_ID"),
                trim(col("ORIGIN_CITY_MARKET_ID")).cast(IntegerType()).alias("ORIGIN_CITY_MARKET_ID"),
                trim(col("ORIGIN_STATE_NM")).cast(IntegerType()).alias("ORIGIN_STATE_NM"),
                trim(col("DEST_AIRPORT_ID")).cast(IntegerType()).alias("DEST_AIRPORT_ID"),
                trim(col("DEST_CITY_MARKET_ID")).cast(IntegerType()).alias("DEST_CITY_MARKET_ID"),
                trim(col("DEST_STATE_NM")).cast(IntegerType()).alias("DEST_STATE_NM"),
                trim(col("CRS_DEP_TIME")).cast(IntegerType()).alias("CRS_DEP_TIME"),
                trim(col("DEP_TIME")).cast(IntegerType()).alias("DEP_TIME"),
                trim(col("DEP_DELAY")).cast(IntegerType()).alias("DEP_DELAY"),
                trim(col("DEP_DELAY_NEW")).cast(IntegerType()).alias("DEP_DELAY_NEW"),
                trim(col("DEP_DEL15")).cast(IntegerType()).alias("DEP_DEL15"),
                trim(col("DEP_DELAY_GROUP")).cast(IntegerType()).alias("DEP_DELAY_GROUP"),
                trim(col("DEP_TIME_BLK")).cast(IntegerType()).alias("DEP_TIME_BLK"),
                trim(col("TAXI_OUT")).cast(IntegerType()).alias("TAXI_OUT"),
                trim(col("WHEELS_OFF")).cast(IntegerType()).alias("WHEELS_OFF"),
#                 trim(col("TAXI_IN")).cast(IntegerType()).alias("TAXI_IN"),
                trim(col("CRS_ARR_TIME")).cast(IntegerType()).alias("CRS_ARR_TIME"),
#                 trim(col("ARR_TIME")).cast(IntegerType()).alias("ARR_TIME"),
#                 trim(col("ARR_DELAY")).cast(IntegerType()).alias("ARR_DELAY"),
#                 trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
#                 trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"), 
                trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("label"), #COLUMNA A PREDECIR
                trim(col("ARR_TIME_BLK")).cast(IntegerType()).alias("ARR_TIME_BLK"),
                trim(col("CRS_ELAPSED_TIME_numero")).cast(IntegerType()).alias("CRS_ELAPSED_TIME_numero"), 
#                 trim(col("ACTUAL_ELAPSED_TIME_numero")).cast(IntegerType()).alias("ACTUAL_ELAPSED_TIME_numero"),
                trim(col("DISTANCE_GROUP")).cast(IntegerType()).alias("DISTANCE_GROUP"),
                trim(col("WEATHER_DELAY")).cast(IntegerType()).alias("WEATHER_DELAY"),
                trim(col("N_FLIGHTS_DAY_ORIGIN")).cast(IntegerType()).alias("N_FLIGHTS_DAY_ORIGIN"),
                trim(col("N_FLIGHTS_DAY_DEST")).cast(IntegerType()).alias("N_FLIGHTS_DAY_DEST"),
                trim(col("N_FLIGHTS_HOUR_ORIGIN")).cast(IntegerType()).alias("N_FLIGHTS_HOUR_ORIGIN"),
                trim(col("N_FLIGHTS_HOUR_DEST")).cast(IntegerType()).alias("N_FLIGHTS_HOUR_DEST"),
                trim(col("ORDER_FLIGHT_ORIGIN")).cast(IntegerType()).alias("ORDER_FLIGHT_ORIGIN"),
                trim(col("DEP_DELAY_CUM")).cast(IntegerType()).alias("DEP_DELAY_CUM"),
                trim(col("ORDER_FLIGHT_DEST")).cast(IntegerType()).alias("ORDER_FLIGHT_DEST"),
                trim(col("ARR_DELAY_CUM")).cast(IntegerType()).alias("ARR_DELAY_CUM"),
                trim(col("ARR_DELAY_FNUMBER_BEFORE")).cast(IntegerType()).alias("ARR_DELAY_FNUMBER_BEFORE"),
                trim(col("CUM_ARR_FNUMBER_DELAY")).cast(IntegerType()).alias("CUM_ARR_FNUMBER_DELAY"),
                trim(col("DEP_DELAY_FNUMBER_BEFORE")).cast(IntegerType()).alias("DEP_DELAY_FNUMBER_BEFORE"),
                trim(col("CUM_DEP_FNUMBER_DELAY")).cast(IntegerType()).alias("CUM_DEP_FNUMBER_DELAY")                        
               )

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer

#Generamos un vector con la columna label  y la columna array features
ignore = ['label']
assembler = VectorAssembler(inputCols=[x for x in df.columns if x not in ignore], 

outputCol='features_without_norm')
df = assembler.transform(df).select(["MONTH", 'label', 'features_without_norm']) 

# COMMAND ----------

# Normalizamos los datos 
normalizer = Normalizer(inputCol="features_without_norm", outputCol="features")
df_normalized = normalizer.transform(df).select(["MONTH", 'label', 'features'])

# COMMAND ----------

# Definimos nuestros datos de entrenamiento del modelo ("train") y los de prediccion ("test")
train=df_normalized.where(df.MONTH!=12)
test=df_normalized.where(df.MONTH==12)

#Dividimos el archivo en train a su vez en: train(90%) y evaluation(10%)
(train, evaluation) = train.randomSplit((0.9, 0.1))

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression, OneVsRest
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler


train_LP = train.select(['label', 'features']) 
evaluation_LP = evaluation.select(['label', 'features'])

#Definimos el algoritmo del modelo (OvR)
# instantiate the base classifier. Only LogisticRegression and NaiveBayes are supported
lr = LogisticRegression(maxIter=10, tol=1E-6, fitIntercept=True) # maxIter=50, regParam=0.05, elasticNetParam=0.05 # elasticNetParam=0.1
# instantiate the One Vs Rest Classifier. 
ovr = OneVsRest(classifier=lr)

# Fit the model
# train the multiclass model.
ovrModel = ovr.fit(train_LP)

# Make predictions.
# score the model on test data.
predictions = ovrModel.transform(evaluation_LP)

# Select (prediction, true label) and compute evaluation error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = " + "%g" % (1.0 - accuracy))

# COMMAND ----------

#Predecimos los grupos de retraso en el mes de diciembre (test)
#Generamos un vector con la columna array features

test_LP = test.select(["label",'features'])

prediccion_december=model_multiclase.transform(test_LP).select(['label','prediction'])
# display(prediccion_december)

# COMMAND ----------

pd_december=prediccion_december.toPandas()

# COMMAND ----------

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pd_december["label"], pd_december["prediction"])
print("Accuracy = %g" %(accuracy*100))

from sklearn.metrics import classification_report
report = classification_report(pd_december["label"], pd_december["prediction"], digits=4)
print(report)

from sklearn.metrics import cohen_kappa_score
cohen_kappa_score = cohen_kappa_score(pd_december["label"], pd_december["prediction"])
print("Cohens Kappa = %g" %cohen_kappa_score)

# COMMAND ----------

#FIN
