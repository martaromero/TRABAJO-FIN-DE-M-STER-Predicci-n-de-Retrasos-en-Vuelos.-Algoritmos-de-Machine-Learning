# REGRESION

# COMMAND ----------

#PARTE1: DEFINIMOS LOS TIPOS DE DATOS DEL ARCHIVO TRAIN 
#Leemos el archivo csv con las cabeceras, importandolo primero todo como string
df = sqlContext.read.format("csv").option("header", "true").load("dbfs:/dataset/datos_preprocesados.csv")

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
                trim(col("ARR_DELAY")).cast(IntegerType()).alias("label"), #COLUMNA A PREDECIR
#                 trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
#                 trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"),
#                 trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
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

# display(df)

# COMMAND ----------

# Definimos nuestros datos de entrenamiento del modelo ("train") y los de prediccion ("test")
train=df.where(df.MONTH!=12)
test=df.where(df.MONTH==12)
#Dividimos el archivo en train a su vez en: train(90%) y evaluation(10%)
(train, evaluation) = train.randomSplit((0.9, 0.1))

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

#Generamos un vector con la columna label  y la columna array features
ignore = ['label']
assembler = VectorAssembler(inputCols=[x for x in train.columns if x not in ignore], outputCol='features')
train_LP = assembler.transform(train).select(['label', 'features']) 
evaluation_LP = assembler.transform(evaluation).select(['label', 'features'])

#Definimos el algoritmo del modelo (decision tree)
model_regresion = DecisionTreeRegressor(labelCol="label", featuresCol="features", maxDepth=14, maxBins=64)

# Fit the model
model_regresion = model_regresion.fit(train_LP)

# Make predictions.
predictions = model_regresion.transform(evaluation_LP)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# COMMAND ----------

#Generamos un vector con la columna array features

ignore=["label"]
assembler = VectorAssembler(inputCols=[x for x in test.columns if x not in ignore], outputCol='features')
test_LP = assembler.transform(test).select(["label",'features'])

# sameModel_multiclase = model_multiclase.load("dbfs:/dataset/modelo_multiclase_DT")
prediccion_december=model_regresion.transform(test_LP).select(['label','prediction'])
display(prediccion_december)

# COMMAND ----------

prediccion_december_pd=prediccion_december.toPandas()

# COMMAND ----------

array_label=prediccion_december_pd.iloc[:,0].values
array_prediction=prediccion_december_pd.iloc[:,1].values

# COMMAND ----------

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from math import sqrt

explained_variance_score=explained_variance_score(array_label, array_prediction)
mean_absolute_error=mean_absolute_error(array_label, array_prediction)
mean_squared_error=mean_squared_error(array_label, array_prediction)
# root_Mean_Squared_Error= sqrt(mean_squared_error(array_label, array_prediction))
median_absolute_error=median_absolute_error(array_label, array_prediction)
r2_score=r2_score(array_label, array_prediction)

print("Explained variance = %s" %explained_variance_score)
print("MAE = %g" %mean_absolute_error)
print("MSE = %g" %mean_squared_error)
print("RMSE = %s" % (sqrt(mean_squared_error)))
print("MedAE = %g" %median_absolute_error)
print("R-squared = %s" %(r2_score*100) + " %")

# COMMAND ----------

# REGRESION ESTANDARIZANDO DATOS

# COMMAND ----------

#PARTE1: DEFINIMOS LOS TIPOS DE DATOS DEL ARCHIVO TRAIN 
#Leemos el archivo csv con las cabeceras, importandolo primero todo como string
df = sqlContext.read.format("csv").option("header", "true").load("dbfs:/dataset/datos_preprocesados.csv")

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
                trim(col("ARR_DELAY")).cast(IntegerType()).alias("label"), #COLUMNA A PREDECIR
#                 trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
#                 trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"),
#                 trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
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

# display(df)

# COMMAND ----------

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler

#Generamos un vector con la columna label  y la columna array features
ignore = ['label']
assembler = VectorAssembler(inputCols=[x for x in df.columns if x not in ignore], outputCol='features_without_stand')
df = assembler.transform(df).select(["MONTH", 'label', 'features_without_stand']) 

scaler1 = StandardScaler(inputCol="features_without_stand", outputCol="scaledFeatures", withStd=True, withMean=False)
# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler1.fit(df)

# Normalize each feature to have unit standard deviation.
scaledData1 = scalerModel.transform(df)

# COMMAND ----------

# Definimos nuestros datos de entrenamiento del modelo ("train") y los de prediccion ("test")
train=scaledData1.where(scaledData1.MONTH!=12)
test=scaledData1.where(scaledData1.MONTH==12)
#Dividimos el archivo en train a su vez en: train(90%) y evaluation(10%)
(train, evaluation) = train.randomSplit((0.9, 0.1))

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

#Generamos un vector con la columna label  y la columna array features
train_LP = train.select(["label", "scaledFeatures"])
evaluation_LP = evaluation.select(['label', 'scaledFeatures'])

#Definimos el algoritmo del modelo (decision tree)
model_regresion = DecisionTreeRegressor(labelCol="label", featuresCol="scaledFeatures", maxDepth=14, maxBins=64)

# Fit the model
model_regresion = model_regresion.fit(train_LP)

# Make predictions.
predictions = model_regresion.transform(evaluation_LP)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# COMMAND ----------

#Generamos un vector con la columna array features

test_LP = test.select(["label",'scaledFeatures'])

# sameModel_multiclase = model_multiclase.load("dbfs:/dataset/modelo_multiclase_DT")
prediccion_december=model_regresion.transform(test_LP).select(['label','prediction'])
display(prediccion_december)

# COMMAND ----------

prediccion_december_pd=prediccion_december.toPandas()

# COMMAND ----------

array_label=prediccion_december_pd.iloc[:,0].values
array_prediction=prediccion_december_pd.iloc[:,1].values

# COMMAND ----------

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from math import sqrt

explained_variance_score=explained_variance_score(array_label, array_prediction)
mean_absolute_error=mean_absolute_error(array_label, array_prediction)
mean_squared_error=mean_squared_error(array_label, array_prediction)
# root_Mean_Squared_Error= sqrt(mean_squared_error(array_label, array_prediction))
median_absolute_error=median_absolute_error(array_label, array_prediction)
r2_score=r2_score(array_label, array_prediction)

print("Explained variance = %s" %explained_variance_score)
print("MAE = %g" %mean_absolute_error)
print("MSE = %g" %mean_squared_error)
print("RMSE = %s" % (sqrt(mean_squared_error)))
print("MedAE = %g" %median_absolute_error)
print("R-squared = %s" %(r2_score*100) + " %")

# COMMAND ----------

# ESTANDARIZACION 2

# COMMAND ----------

#PARTE1: DEFINIMOS LOS TIPOS DE DATOS DEL ARCHIVO TRAIN 
#Leemos el archivo csv con las cabeceras, importandolo primero todo como string
df = sqlContext.read.format("csv").option("header", "true").load("dbfs:/dataset/datos_preprocesados.csv")

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
                trim(col("ARR_DELAY")).cast(IntegerType()).alias("label"), #COLUMNA A PREDECIR
#                 trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
#                 trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"),
#                 trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
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

# display(df)

# COMMAND ----------

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import VectorAssembler

#Generamos un vector con la columna label  y la columna array features
ignore = ['label']
assembler = VectorAssembler(inputCols=[x for x in df.columns if x not in ignore], outputCol='features_without_stand')
df = assembler.transform(df).select(["MONTH", 'label', 'features_without_stand']) 

scaler1 = StandardScaler(inputCol="features_without_stand", outputCol="scaledFeatures", withStd=True, withMean=True)
# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler1.fit(df)

# Normalize each feature to have unit standard deviation.
scaledData1 = scalerModel.transform(df)

# COMMAND ----------

# Definimos nuestros datos de entrenamiento del modelo ("train") y los de prediccion ("test")
train=scaledData1.where(scaledData1.MONTH!=12)
test=scaledData1.where(scaledData1.MONTH==12)
#Dividimos el archivo en train a su vez en: train(90%) y evaluation(10%)
(train, evaluation) = train.randomSplit((0.9, 0.1))

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler

#Generamos un vector con la columna label  y la columna array features
train_LP = train.select(["label", "scaledFeatures"])
evaluation_LP = evaluation.select(['label', 'scaledFeatures'])

#Definimos el algoritmo del modelo (decision tree)
model_regresion = DecisionTreeRegressor(labelCol="label", featuresCol="scaledFeatures", maxDepth=14, maxBins=64)

# Fit the model
model_regresion = model_regresion.fit(train_LP)

# Make predictions.
predictions = model_regresion.transform(evaluation_LP)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

# COMMAND ----------

#Generamos un vector con la columna array features

test_LP = test.select(["label",'scaledFeatures'])
prediccion_december=model_regresion.transform(test_LP).select(['label','prediction'])
display(prediccion_december)

# COMMAND ----------

prediccion_december_pd=prediccion_december.toPandas()

# COMMAND ----------

array_label=prediccion_december_pd.iloc[:,0].values
array_prediction=prediccion_december_pd.iloc[:,1].values

# COMMAND ----------

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
from math import sqrt

explained_variance_score=explained_variance_score(array_label, array_prediction)
mean_absolute_error=mean_absolute_error(array_label, array_prediction)
mean_squared_error=mean_squared_error(array_label, array_prediction)
# root_Mean_Squared_Error= sqrt(mean_squared_error(array_label, array_prediction))
median_absolute_error=median_absolute_error(array_label, array_prediction)
r2_score=r2_score(array_label, array_prediction)

print("Explained variance = %s" %explained_variance_score)
print("MAE = %g" %mean_absolute_error)
print("MSE = %g" %mean_squared_error)
print("RMSE = %s" % (sqrt(mean_squared_error)))
print("MedAE = %g" %median_absolute_error)
print("R-squared = %s" %(r2_score*100) + " %")
