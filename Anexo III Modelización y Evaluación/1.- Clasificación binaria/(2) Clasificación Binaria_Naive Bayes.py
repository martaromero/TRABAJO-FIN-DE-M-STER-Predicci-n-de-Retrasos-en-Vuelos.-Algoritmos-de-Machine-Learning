# Databricks notebook source
# PRIMERA PARTE: CLASIFICACION BINARIA (0: vuelo en hora, 1: vuelo retrasado)

# COMMAND ----------

#PARTE1: DEFINIMOS LOS TIPOS DE DATOS DEL ARCHIVO TRAIN 
#Leemos el archivo csv con las cabeceras, importandolo primero todo como string
df = sqlContext.read.format("csv").option("header", "true").load("dbfs:/dataset/df_preprocesado/part-00000-e613ac01-a7eb-431c-8718-5c854f2e13ce-c000.csv")

# COMMAND ----------

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
#                 trim(col("DEP_DELAY")).cast(IntegerType()).alias("DEP_DELAY"), #Naive Bayes no permite argumentos negativos
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
                trim(col("ARR_DEL15")).cast(IntegerType()).alias("label"), #COLUMNA A PREDECIR
#                 trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
                trim(col("ARR_TIME_BLK")).cast(IntegerType()).alias("ARR_TIME_BLK"),
#                 trim(col("CRS_ELAPSED_TIME_numero")).cast(IntegerType()).alias("CRS_ELAPSED_TIME_numero"), # Naive Bayes no permite argumentos negativos
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

# COMMAND ----------

#Dividimos el archivo en train a su vez en: train(90%) y evaluation(10%)
(train, evaluation) = train.randomSplit((0.9, 0.1))

# COMMAND ----------

# Entrenamos y calibramos el modelo modificando sus parametros internos viendo sus resultados en evaluation

from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

#Definimos el dataset para la prediccion del label ARR_DEL15
#Generamos un vector con la columna label  y la columna array features

ignore = ['label']
assembler = VectorAssembler(inputCols=[x for x in train.columns if x not in ignore], outputCol='features')
train_LP = assembler.transform(train).select(['label', 'features']) 
evaluation_LP = assembler.transform(evaluation).select(['label', 'features'])

#Definimos el algoritmo del modelo (Naive Bayes)
nb = NaiveBayes (smoothing=0.1, modelType="multinomial") 

# Fit the model
model = nb.fit(train_LP)

#Save the model
#model.save("dbfs:/dataset/modelo_binario_NB")

# Make predictions.
predictions = model.transform(evaluation_LP)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Evaluation Error = " + "%g" % (1.0 - accuracy))

#METRICAS

from math import sqrt 

LabelandPreds=predictions.select("label", "prediction")
RDD_LP=LabelandPreds.rdd

fp=RDD_LP.filter(lambda (k, v): k != v and v==1).count() #prediccion 1 y label 0
print ("False Positives = " + "%g" % fp)

fn=RDD_LP.filter(lambda (k, v): k != v and v==0).count() #prediccion 0 y label 1
print ("False Negatives = " + "%g" % fn)

tp=RDD_LP.filter(lambda (k, v): k == v and v==1).count() #prediccion 1 y label 1
print ("True Positives = " + "%g" % tp)

tn=RDD_LP.filter(lambda (k, v): k == v and v==0).count() #prediccion 0 y label 0
print ("True Negatives = " + "%g" % tn)

sensibilidad=(float(tp)/(float(fn)+float(tp)))
print ("Sensivity or true positive rate = " + "%g" %(sensibilidad*100) + "%")

if tn == 0: especificidad=0;
else: especificidad=(float(tn)/(float(tn)+float(fp)));
print ("Specifity or true negative rate = " + "%g" %(especificidad*100) + "%")

if tp ==0: ppv=0;
else: ppv=(float(tp)/(float(tp)+float(fp)));
print("Positive predictive value = " + "%g" %(ppv*100) + "%")

if tn == 0: npv=0;
else: npv=(float(tn)/(float(tn)+float(fn)));
print("Negative predictive value = " + "%g" %(npv*100) + "%")

acc=((float(tp)+float(tn))/(float(tp)+float(tn)+float(fp)+float(fn)))
print("Accuracy = " + "%g" %(acc*100) + "%")

F1=(float(2*tp)/float(2*tp+fp+fn))
print ("F1 Score = " + "%g" %(F1*100) + "%")

E1=(float(2*tn)/float(2*tn+fp+fn))
print ("E1 Score = " + "%g" %(E1*100) + "%")

informedness=float(sensibilidad)+float(especificidad)-1
print ("Informedness = " + "%g" %(informedness*100) + "%")

markedness=ppv+npv-1
print ("Markedness = " + "%g" %(markedness*100) + "%")

hm=2*(F1*E1)/(F1+E1)
print("CP: Harmonic mean F1, E1 = " + "%g" %hm)

MCC=sqrt(informedness*markedness)
print("MCC = " + "%g" %MCC)

if ppv==0: dp=0;
else: dp=3*(sensibilidad*npv*ppv) /((sensibilidad*ppv) + (sensibilidad*npv) + (npv*ppv));
print ("DP = " + "%g" %dp)
# COMMAND ----------

# Predecimos los retrasos en el mes de diciembre (test) con el modelo ya calibrado y entrenado

from pyspark.ml.feature import VectorAssembler
from  pyspark.sql import Column

ignore=["label"]
assembler = VectorAssembler(inputCols=[x for x in test.columns if x not in ignore], outputCol='features')
test_LP = assembler.transform(test).select(["label",'features'])

# COMMAND ----------

# sameModel = model.load("dbfs:/dataset/modelo_binario_LR")
prediccion_december=model.transform(test_LP).select(['label','prediction'])
display(prediccion_december)

# COMMAND ----------

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediccion_december)
print("Evaluation Error = " + "%g" % (1.0 - accuracy))

#METRICAS

from math import sqrt 

LabelandPreds_december=prediccion_december.select("label", "prediction")
RDD_LP_december=LabelandPreds_december.rdd

fp=RDD_LP_december.filter(lambda (k, v): k != v and v==1).count() #prediccion 1 y label 0
print ("False Positives = " + "%g" % fp)

fn=RDD_LP_december.filter(lambda (k, v): k != v and v==0).count() #prediccion 0 y label 1
print ("False Negatives = " + "%g" % fn)

tp=RDD_LP_december.filter(lambda (k, v): k == v and v==1).count() #prediccion 1 y label 1
print ("True Positives = " + "%g" % tp)

tn=RDD_LP_december.filter(lambda (k, v): k == v and v==0).count() #prediccion 0 y label 0
print ("True Negatives = " + "%g" % tn)

sensibilidad=(float(tp)/(float(fn)+float(tp)))
print ("Sensivity or true positive rate = " + "%g" %(sensibilidad*100) + "%")

if tn == 0: especificidad=0;
else: especificidad=(float(tn)/(float(tn)+float(fp)));
print ("Specifity or true negative rate = " + "%g" %(especificidad*100) + "%")

if tp ==0: ppv=0;
else: ppv=(float(tp)/(float(tp)+float(fp)));
print("Positive predictive value = " + "%g" %(ppv*100) + "%")

if tn == 0: npv=0;
else: npv=(float(tn)/(float(tn)+float(fn)));
print("Negative predictive value = " + "%g" %(npv*100) + "%")

acc=((float(tp)+float(tn))/(float(tp)+float(tn)+float(fp)+float(fn)))
print("Accuracy = " + "%g" %(acc*100) + "%")

F1=(float(2*tp)/float(2*tp+fp+fn))
print ("F1 Score = " + "%g" %(F1*100) + "%")

E1=(float(2*tn)/float(2*tn+fp+fn))
print ("E1 Score = " + "%g" %(E1*100) + "%")

informedness=float(sensibilidad)+float(especificidad)-1
print ("Informedness = " + "%g" %(informedness*100) + "%")

markedness=ppv+npv-1
print ("Markedness = " + "%g" %(markedness*100) + "%")

hm=2*(F1*E1)/(F1+E1)
print("CP: Harmonic mean F1, E1 = " + "%g" %hm)

MCC=sqrt(informedness*markedness)
print("MCC = " + "%g" %MCC)

if ppv==0: dp=0;
else: dp=3*(sensibilidad*npv*ppv) /((sensibilidad*ppv) + (sensibilidad*npv) + (npv*ppv));
print ("DP = " + "%g" %dp)

# COMMAND ----------

# CLASIFICACION BINARIA CON DATOS NORMALIZADOS

# COMMAND ----------

#PARTE1: DEFINIMOS LOS TIPOS DE DATOS DEL ARCHIVO TRAIN 
#Leemos el archivo csv con las cabeceras, importandolo primero todo como string
df = sqlContext.read.format("csv").option("header", "true").load("dbfs:/dataset/df_preprocesado/part-00000-e613ac01-a7eb-431c-8718-5c854f2e13ce-c000.csv")

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
#                 trim(col("DEP_DELAY")).cast(IntegerType()).alias("DEP_DELAY"), # no acepta argumentos negativos
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
                trim(col("ARR_DEL15")).cast(IntegerType()).alias("label"), #COLUMNA A PREDECIR
#                 trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
                trim(col("ARR_TIME_BLK")).cast(IntegerType()).alias("ARR_TIME_BLK"),
#                 trim(col("CRS_ELAPSED_TIME_numero")).cast(IntegerType()).alias("CRS_ELAPSED_TIME_numero"), # no acepta argumentos negativos
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
assembler = VectorAssembler(inputCols=[x for x in df.columns if x not in ignore], outputCol='features_without_norm')
df = assembler.transform(df).select(["MONTH", 'label', 'features_without_norm']) 

# Normalizamos los datos 
normalizer = Normalizer(inputCol="features_without_norm", outputCol="features")
df_normalized = normalizer.transform(df).select(["MONTH", 'label', 'features'])

# COMMAND ----------

# Definimos nuestros datos de entrenamiento del modelo ("train") y los de prediccion ("test")
train=df_normalized.where(df.MONTH!=12)
test=df_normalized.where(df.MONTH==12)

# COMMAND ----------

#Dividimos el archivo en train a su vez en: train(90%) y evaluation(10%)
(train, evaluation) = train.randomSplit((0.9, 0.1))

# COMMAND ----------

# Entrenamos y calibramos el modelo (con los datos ya normalizados) modificando sus parametros internos viendo sus resultados en evaluation

from pyspark.ml.classification import NaiveBayes, NaiveBayesModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

#Definimos el dataset para la prediccion del label ARR_DEL15

training= train.select("label", "features")

evaluating= evaluation.select("label", "features")

#Definimos el algoritmo del modelo (Naive Bayes)
nb = NaiveBayes (smoothing=1000.0, modelType="multinomial") 

# Fit the model
model = nb.fit(training)

#Save the model
# model.save("dbfs:/dataset/modelo_binario_NB")

# Make predictions.
predictions = model.transform(evaluating)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Evaluation Error = " + "%g" % (1.0 - accuracy))

#METRICAS

from math import sqrt 

LabelandPreds=predictions.select("label", "prediction")
RDD_LP=LabelandPreds.rdd

fp=RDD_LP.filter(lambda (k, v): k != v and v==1).count() #prediccion 1 y label 0
print ("False Positives = " + "%g" % fp)

fn=RDD_LP.filter(lambda (k, v): k != v and v==0).count() #prediccion 0 y label 1
print ("False Negatives = " + "%g" % fn)

tp=RDD_LP.filter(lambda (k, v): k == v and v==1).count() #prediccion 1 y label 1
print ("True Positives = " + "%g" % tp)

tn=RDD_LP.filter(lambda (k, v): k == v and v==0).count() #prediccion 0 y label 0
print ("True Negatives = " + "%g" % tn)

sensibilidad=(float(tp)/(float(fn)+float(tp)))
print ("Sensivity or true positive rate = " + "%g" %(sensibilidad*100) + "%")

if tn == 0: especificidad=0;
else: especificidad=(float(tn)/(float(tn)+float(fp)));
print ("Specifity or true negative rate = " + "%g" %(especificidad*100) + "%")

if tp ==0: ppv=0;
else: ppv=(float(tp)/(float(tp)+float(fp)));
print("Positive predictive value = " + "%g" %(ppv*100) + "%")

if tn == 0: npv=0;
else: npv=(float(tn)/(float(tn)+float(fn)));
print("Negative predictive value = " + "%g" %(npv*100) + "%")

acc=((float(tp)+float(tn))/(float(tp)+float(tn)+float(fp)+float(fn)))
print("Accuracy = " + "%g" %(acc*100) + "%")

F1=(float(2*tp)/float(2*tp+fp+fn))
print ("F1 Score = " + "%g" %(F1*100) + "%")

E1=(float(2*tn)/float(2*tn+fp+fn))
print ("E1 Score = " + "%g" %(E1*100) + "%")

informedness=float(sensibilidad)+float(especificidad)-1
print ("Informedness = " + "%g" %(informedness*100) + "%")

markedness=ppv+npv-1
print ("Markedness = " + "%g" %(markedness*100) + "%")

hm=2*(F1*E1)/(F1+E1)
print("CP: Harmonic mean F1, E1 = " + "%g" %hm)

MCC=sqrt(informedness*markedness)
print("MCC = " + "%g" %MCC)

if ppv==0: dp=0;
else: dp=3*(sensibilidad*npv*ppv) /((sensibilidad*ppv) + (sensibilidad*npv) + (npv*ppv));
print ("DP = " + "%g" %dp)

# COMMAND ----------

# Predecimos los retrasos en el mes de diciembre (test) con el modelo ya calibrado y entrenado

from  pyspark.sql import Column

testing = test.select("label", "features")

prediccion_december=model.transform(testing).select(['label','prediction'])
# display(prediccion_december)

# COMMAND ----------

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(prediccion_december)
print("Evaluation Error = " + "%g" % (1.0 - accuracy))

#METRICAS

from math import sqrt 

LabelandPreds_december=prediccion_december.select("label", "prediction")
RDD_LP_december=LabelandPreds_december.rdd

fp=RDD_LP_december.filter(lambda (k, v): k != v and v==1).count() #prediccion 1 y label 0
print ("False Positives = " + "%g" % fp)

fn=RDD_LP_december.filter(lambda (k, v): k != v and v==0).count() #prediccion 0 y label 1
print ("False Negatives = " + "%g" % fn)

tp=RDD_LP_december.filter(lambda (k, v): k == v and v==1).count() #prediccion 1 y label 1
print ("True Positives = " + "%g" % tp)

tn=RDD_LP_december.filter(lambda (k, v): k == v and v==0).count() #prediccion 0 y label 0
print ("True Negatives = " + "%g" % tn)

sensibilidad=(float(tp)/(float(fn)+float(tp)))
print ("Sensivity or true positive rate = " + "%g" %(sensibilidad*100) + "%")

if tn == 0: especificidad=0;
else: especificidad=(float(tn)/(float(tn)+float(fp)));
print ("Specifity or true negative rate = " + "%g" %(especificidad*100) + "%")

if tp ==0: ppv=0;
else: ppv=(float(tp)/(float(tp)+float(fp)));
print("Positive predictive value = " + "%g" %(ppv*100) + "%")

if tn == 0: npv=0;
else: npv=(float(tn)/(float(tn)+float(fn)));
print("Negative predictive value = " + "%g" %(npv*100) + "%")

acc=((float(tp)+float(tn))/(float(tp)+float(tn)+float(fp)+float(fn)))
print("Accuracy = " + "%g" %(acc*100) + "%")

F1=(float(2*tp)/float(2*tp+fp+fn))
print ("F1 Score = " + "%g" %(F1*100) + "%")

E1=(float(2*tn)/float(2*tn+fp+fn))
print ("E1 Score = " + "%g" %(E1*100) + "%")

informedness=float(sensibilidad)+float(especificidad)-1
print ("Informedness = " + "%g" %(informedness*100) + "%")

markedness=ppv+npv-1
print ("Markedness = " + "%g" %(markedness*100) + "%")

hm=2*(F1*E1)/(F1+E1)
print("CP: Harmonic mean F1, E1 = " + "%g" %hm)

MCC=sqrt(informedness*markedness)
print("MCC = " + "%g" %MCC)

if ppv==0: dp=0;
else: dp=3*(sensibilidad*npv*ppv) /((sensibilidad*ppv) + (sensibilidad*npv) + (npv*ppv));
print ("DP = " + "%g" %dp)
