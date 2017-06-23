# Databricks notebook source
# MAGIC %md ### PREPROCESAMIENTO DE LOS DATOS

# COMMAND ----------

#Vemos el directorio donde tenemos guardado nuestro archivo train_ver2.csv

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/dataset/"))

# COMMAND ----------

#PARTE1: DEFINIMOS LOS TIPOS DE DATOS DEL ARCHIVO TRAIN 
#Leemos el archivo csv con las cabeceras, importandolo primero todo como string
df = sqlContext.read.format("csv").option("header", "true").load("dbfs:/dataset/datos_filtrados_2016.csv")
display(df)

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import trim, col

#Quitamos los espacios de cada una de las columnas con (trim(col("x"))) y definimos ya el tipo de dato en cada una de ellas (.cast(DateType())) con su nombre (.alias("x"))

df1 = df.select(trim(col("MONTH")).cast(IntegerType()).alias("MONTH"),
                trim(col("DAY_OF_MONTH")).cast(IntegerType()).alias("DAY_OF_MONTH"),
                trim(col("FL_DATE")).cast(StringType()).alias("FL_DATE"),
                trim(col("DAY_OF_WEEK")).cast(IntegerType()).alias("DAY_OF_WEEK"),
                trim(col("UNIQUE_CARRIER")).cast(StringType()).alias("UNIQUE_CARRIER"),
                trim(col("TAIL_NUM")).cast(StringType()).alias("TAIL_NUM"),
                trim(col("FL_NUM")).cast(IntegerType()).alias("FL_NUM"),
                trim(col("ORIGIN_AIRPORT_ID")).cast(IntegerType()).alias("ORIGIN_AIRPORT_ID"),
                trim(col("ORIGIN_CITY_MARKET_ID")).cast(IntegerType()).alias("ORIGIN_CITY_MARKET_ID"),
                trim(col("ORIGIN_STATE_NM")).cast(StringType()).alias("ORIGIN_STATE_NM"),
                trim(col("DEST_AIRPORT_ID")).cast(IntegerType()).alias("DEST_AIRPORT_ID"),
                trim(col("DEST_CITY_MARKET_ID")).cast(IntegerType()).alias("DEST_CITY_MARKET_ID"),
                trim(col("DEST_STATE_ABR")).cast(StringType()).alias("DEST_STATE_ABR"),
                trim(col("CRS_DEP_TIME")).cast(IntegerType()).alias("CRS_DEP_TIME"),
                trim(col("DEP_TIME")).cast(IntegerType()).alias("DEP_TIME"),
                trim(col("DEP_DELAY")).cast(IntegerType()).alias("DEP_DELAY"),
                trim(col("DEP_DELAY_NEW")).cast(IntegerType()).alias("DEP_DELAY_NEW"),
                trim(col("DEP_DEL15")).cast(IntegerType()).alias("DEP_DEL15"),
                trim(col("DEP_DELAY_GROUP")).cast(IntegerType()).alias("DEP_DELAY_GROUP"),
                trim(col("DEP_TIME_BLK")).cast(StringType()).alias("DEP_TIME_BLK"),
                trim(col("TAXI_OUT")).cast(IntegerType()).alias("TAXI_OUT"),
                trim(col("WHEELS_OFF")).cast(IntegerType()).alias("WHEELS_OFF"),
                trim(col("TAXI_IN")).cast(IntegerType()).alias("TAXI_IN"),
                trim(col("CRS_ARR_TIME")).cast(IntegerType()).alias("CRS_ARR_TIME"),
                trim(col("ARR_TIME")).cast(IntegerType()).alias("ARR_TIME"),
                trim(col("ARR_DELAY")).cast(IntegerType()).alias("ARR_DELAY"),
                trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
                trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"),
                trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
                trim(col("ARR_TIME_BLK")).cast(StringType()).alias("ARR_TIME_BLK"),
                trim(col("CANCELLED")).cast(IntegerType()).alias("CANCELLED"),
                trim(col("DIVERTED")).cast(IntegerType()).alias("DIVERTED"),
                trim(col("CRS_ELAPSED_TIME")).cast(IntegerType()).alias("CRS_ELAPSED_TIME"),
                trim(col("ACTUAL_ELAPSED_TIME")).cast(IntegerType()).alias("ACTUAL_ELAPSED_TIME"),
                trim(col("DISTANCE")).cast(IntegerType()).alias("DISTANCE"),
                trim(col("DISTANCE_GROUP")).cast(IntegerType()).alias("DISTANCE_GROUP"),
                trim(col("WEATHER_DELAY")).cast(IntegerType()).alias("WEATHER_DELAY")
               )
display(df1)

# COMMAND ----------

vuelos_totales = df1.count()
print("Vuelos totales durante 2016 = %g" %vuelos_totales)
cancelados = df1.where(col("CANCELLED")==1).count()
print("Vuelos cancelados durante 2016 = %g" %cancelados)
derivados = df1.where(col("DIVERTED")==1).count()
print("Vuelos derivados durante 2016 = %g" %derivados)

# COMMAND ----------

#Comprobamos que las columnas son efectivamente como hemos definido
df1.printSchema()

# COMMAND ----------

df1.registerTempTable("df1")

# COMMAND ----------

# MAGIC %sql select distinct ORIGIN_STATE_NM from df1 order by ORIGIN_STATE_NM asc

# COMMAND ----------

# Convertimos la columna ORIGIN_STATE_NM y DEST_STATE_ABR al mismo formato de texto

def categorize_destination_airport (s) :

  if s  == 'AB' : return 'Alberta, Canada';
  elif s  == 'AK' : return 'Alaska';
  elif s  == 'AL' : return 'Alabama';
  elif s  == 'AR' : return 'Arkansas';
  elif s  == 'AZ' : return 'Arizona';
  elif s  == 'BC' : return 'British Columbia, Canada';
  elif s  == 'CA' : return 'California';
  elif s  == 'CO' : return 'Colorado';
  elif s  == 'CT' : return 'Connecticut';
  elif s  == 'DC' : return 'District of Columbia';
  elif s  == 'DE' : return 'Delaware';
  elif s  == 'FL' : return 'Florida';
  elif s  == 'GA' : return 'Georgia';
  elif s  == 'HI' : return 'Hawaii';
  elif s  == 'IA' : return 'Iowa';
  elif s  == 'ID' : return 'Idaho';
  elif s  == 'IL' : return 'Illinois';
  elif s  == 'IN' : return 'Indiana';
  elif s  == 'KS' : return 'Kansas';
  elif s  == 'KY' : return 'Kentucky';
  elif s  == 'LA' : return 'Louisiana';
  elif s  == 'MA' : return 'Massachusetts';
  elif s  == 'MB' : return 'Manitoba, Canada';
  elif s  == 'MD' : return 'Maryland';
  elif s  == 'ME' : return 'Maine';
  elif s  == 'MI' : return 'Michigan';
  elif s  == 'MN' : return 'Minnesota';
  elif s  == 'MO' : return 'Missouri';
  elif s  == 'MS' : return 'Mississippi';
  elif s  == 'MT' : return 'Montana';
  elif s  == 'NB' : return 'New Brunswick, Canada';
  elif s  == 'NC' : return 'North Carolina';
  elif s  == 'ND' : return 'North Dakota';
  elif s  == 'NE' : return 'Nebraska';
  elif s  == 'NH' : return 'New Hampshire';
  elif s  == 'NJ' : return 'New Jersey';
  elif s  == 'NL' : return 'Newfoundland and Labrador, Canada';
  elif s  == 'NM' : return 'New Mexico';
  elif s  == 'NS' : return 'Nova Scotia, Canada';
  elif s  == 'NT' : return 'Northwest Territories, Canada';
  elif s  == 'NU' : return 'Nunavut Territory, Canada';
  elif s  == 'NV' : return 'Nevada';
  elif s  == 'NY' : return 'New York';
  elif s  == 'OH' : return 'Ohio';
  elif s  == 'OK' : return 'Oklahoma';
  elif s  == 'ON' : return 'Ontario, Canada';
  elif s  == 'OR' : return 'Oregon';
  elif s  == 'PA' : return 'Pennsylvania';
  elif s  == 'PE' : return 'Prince Edward Island, Canada';
  elif s  == 'PR' : return 'Puerto Rico';
  elif s  == 'QC' : return 'Quebec, Canada';
  elif s  == 'RI' : return 'Rhode Island';
  elif s  == 'SC' : return 'South Carolina';
  elif s  == 'SD' : return 'South Dakota';
  elif s  == 'SK' : return 'Saskatchewan, Canada';
  elif s  == 'TN' : return 'Tennessee';
  elif s  == 'TT' : return 'U.S. Pacific Trust Territories and Possessions';
  elif s  == 'TX' : return 'Texas';
  elif s  == 'UT' : return 'Utah';
  elif s  == 'VA' : return 'Virginia';
  elif s  == 'VI' : return 'U.S. Virgin Islands';
  elif s  == 'VT' : return 'Vermont';
  elif s  == 'WA' : return 'Washington';
  elif s  == 'WI' : return 'Wisconsin';
  elif s  == 'WV' : return 'West Virginia';
  elif s  == 'WY' : return 'Wyoming';
  elif s  == 'YT' : return 'Yukon Territory, Canada';
  
udf_categorize_destination_airport  = udf(categorize_destination_airport, StringType())
df1_correct = df1.withColumn("DEST_STATE_NM", udf_categorize_destination_airport("DEST_STATE_ABR"))
display(df1_correct)

# COMMAND ----------

# Filtramos para quitarnos los vuelos cancelados y derivados a otros aeropuertos. A continuacion sustituimos campos nulls por 0
df_filter1=df1_correct.where(col("CANCELLED")!=1)
df_filter2=df_filter1.where(col("DIVERTED")!=1)
df_filter2=df_filter2.na.fill(0)
display(df_filter2)

# COMMAND ----------

# Nos creamos columnas adicionales -> nº vuelos el mismo dia en O/D, numero de vuelos en la misma franja horaria en O/D, orden de vuelo en el dia, numero de vuelo (FL_NUM) ya retrasado a lo largo del dia

# COLUMNA ADICIONAL 1: Vuelos en periodos Festivos

def categorize_festividad (s) :
#   Navidad
  if s == "2016-01-01": return 1;
  elif s =="2016-01-02": return 1;
  elif s =="2016-01-03": return 1;
  elif s =="2016-01-04": return 1;
  elif s =="2016-01-05": return 1;
#   Martin Luther King
  elif s =="2016-01-13": return 1;
  elif s =="2016-01-14": return 1;
  elif s =="2016-01-15": return 1;
  elif s =="2016-01-16": return 1;
  elif s =="2016-01-17": return 1;
  elif s =="2016-01-18": return 1;
  elif s =="2016-01-19": return 1;
  elif s =="2016-01-20": return 1;
#   Dia del presidente
  elif s =="2016-02-10": return 1;
  elif s =="2016-02-11": return 1;
  elif s =="2016-02-12": return 1;
  elif s =="2016-02-13": return 1;
  elif s =="2016-02-14": return 1;
  elif s =="2016-02-15": return 1;
  elif s =="2016-02-16": return 1;
  elif s =="2016-02-17": return 1;
#   Pascua
  elif s =="2016-03-18": return 1;
  elif s =="2016-03-19": return 1;
  elif s =="2016-03-20": return 1;
  elif s =="2016-03-21": return 1;
  elif s =="2016-03-22": return 1;
  elif s =="2016-03-23": return 1;
  elif s =="2016-03-24": return 1;
  elif s =="2016-03-25": return 1;
  elif s =="2016-03-26": return 1;
  elif s =="2016-03-27": return 1;
  elif s =="2016-03-28": return 1;
  elif s =="2016-03-29": return 1;
#   Memorial day
  elif s =="2016-05-25": return 1;
  elif s =="2016-05-26": return 1;
  elif s =="2016-05-27": return 1;
  elif s =="2016-05-28": return 1;
  elif s =="2016-05-29": return 1;
  elif s =="2016-05-30": return 1;
  elif s =="2016-05-31": return 1;
  elif s =="2016-05-01": return 1;
# Dia de la independencia
  elif s =="2016-06-29": return 1;
  elif s =="2016-06-30": return 1;
  elif s =="2016-07-01": return 1;
  elif s =="2016-07-02": return 1;
  elif s =="2016-07-03": return 1;
  elif s =="2016-07-04": return 1;
  elif s =="2016-07-05": return 1;
  elif s =="2016-07-06": return 1;
# Dia del trabajo
  elif s =="2016-08-31": return 1;
  elif s =="2016-09-01": return 1;
  elif s =="2016-09-02": return 1;
  elif s =="2016-09-03": return 1;
  elif s =="2016-09-04": return 1;
  elif s =="2016-09-05": return 1;
  elif s =="2016-09-06": return 1;
  elif s =="2016-09-07": return 1;
# Dia de los veteranos
  elif s =="2016-11-09": return 1;
  elif s =="2016-11-10": return 1;
  elif s =="2016-11-11": return 1;
  elif s =="2016-11-12": return 1;
  elif s =="2016-11-13": return 1;
  elif s =="2016-11-14": return 1;
  elif s =="2016-11-15": return 1;
# Dia de accion de gracias
  elif s =="2016-11-21": return 1;
  elif s =="2016-11-22": return 1;
  elif s =="2016-11-23": return 1;
  elif s =="2016-11-24": return 1;
  elif s =="2016-11-25": return 1;
  elif s =="2016-11-26": return 1;
  elif s =="2016-11-27": return 1;
  elif s =="2016-11-28": return 1;
  
  else: return 0;
  
udf_categorize_festividad  = udf(categorize_festividad, IntegerType())
df2 = df_filter2.withColumn("HOLIDAYS", udf_categorize_festividad("FL_DATE"))
display(df2)

# COMMAND ----------

# Para crearnos la columna adicional 2, necesitamos agrupar por fecha y aeropuerto de origen. 
# (1) Para ello nos crearemos primeramente un codigo unico para cada fila, concatenando la fecha y el aeropuerto de origen.
# (2) A continuacion realizaremos una vista a traves de la creacion de un nuevo dataframe que agrupe y cuente el numero de vuelos con el mismo codigo (fecha y aeropuerto de origen)
# (3) Sera necesario unir el dataframe primario con la vista realizada del numero de vuelos a traves de una columna en comun, en este caso el codigo unico creado en el paso (1)

# COMMAND ----------

# (1) Creamos tabla temporal para poder hacer la concatenacion posterior de la fecha y el aeropuerto de origen
df2.registerTempTable("df2")

# COMMAND ----------

# (1) Creamos el codigo unico de cada fila mediante la concatenacion de la fecha y el aeropuerto de origen
df3=sqlContext.sql("SELECT *, CONCAT(FL_DATE,',',ORIGIN_AIRPORT_ID) as DATE_ORIGIN FROM df2")
display(df3)

# COMMAND ----------

# (2) Creamos la tabla temporal para poder hacer la consulta y obtener la vista deseada
df3.registerTempTable("df3")

# COMMAND ----------

# (2) Realizamos la consulta para obtener el numero de vuelos al dia en el aeropuerto de origen
df4=sqlContext.sql("SELECT DATE_ORIGIN, COUNT(DATE_ORIGIN) as N_FLIGHTS_DAY_ORIGIN FROM df3 GROUP BY DATE_ORIGIN")
display(df4)

# COMMAND ----------

# (3) Unimos los dataframes df3 y df4 para obtener la columna con el numero de vuelos diarios en el aeropuerto de origen. 
# Creacion de la COLUMNA ADICIONAL 2
joined1=df3.join(df4, "DATE_ORIGIN", 'inner')
display(joined1)

# COMMAND ----------

# Repetimos los pasos para la creacion de la COLUMNA ADICIONAL 3, pero esta vez con el aeropuerto de destino.

# COMMAND ----------

# (1) Creamos tabla temporal para poder hacer la concatenacion posterior de la fecha y el aeropuerto de destino
joined1.registerTempTable("joined1")
# (1) Creamos el codigo unico de cada fila mediante la concatenacion de la fecha y el aeropuerto de destino
df5=sqlContext.sql("SELECT *, CONCAT(FL_DATE,',',DEST_AIRPORT_ID) as DATE_DEST FROM joined1")
# (2) Creamos la tabla temporal para poder hacer la consulta y obtener la vista deseada
df5.registerTempTable("df5")
# (2) Realizamos la consulta para obtener el numero de vuelos al dia en el aeropuerto de destino
df6=sqlContext.sql("SELECT DATE_DEST, COUNT(DATE_DEST) as N_FLIGHTS_DAY_DEST FROM df5 GROUP BY DATE_DEST")
# (3) Unimos los dataframes df5 y df6 para obtener la columna con el numero de vuelos diarios en el aeropuerto de destino. 
# Creacion de la COLUMNA ADICIONAL 3
joined2=df5.join(df6, "DATE_DEST", 'inner')
display(joined2)

# COMMAND ----------

# Creacion de la COLUMNA ADICIONAL 4 y de la COLUMNA ADICIONAL 5, las cuales seran muy parecidas a las anteriores 
# pero viendo el numero de vuelos por dia y por franja horaria en cada aeropuerto de origen y de destino, respectivamente
# COLUMNA ADICIONAL 4
# (1) Creamos tabla temporal para poder hacer la concatenacion posterior de la fecha, franja horaria y el aeropuerto de origen
joined2.registerTempTable("joined2")
# (1) Creamos el codigo unico de cada fila mediante la concatenacion de la fecha, franja horaria y el aeropuerto de origen
df7=sqlContext.sql("SELECT *, CONCAT(FL_DATE,',',ORIGIN_AIRPORT_ID,',',DEP_TIME_BLK) as DATE_HOUR_ORIGIN FROM joined2")
# (2) Creamos la tabla temporal para poder hacer la consulta y obtener la vista deseada
df7.registerTempTable("df7")
# (2) Realizamos la consulta para obtener el numero de vuelos al dia y por franja horaria en el aeropuerto de origen
df8=sqlContext.sql("SELECT DATE_HOUR_ORIGIN, COUNT(DATE_HOUR_ORIGIN) as N_FLIGHTS_HOUR_ORIGIN FROM df7 GROUP BY DATE_HOUR_ORIGIN")
# (3) Unimos los dataframes df7 y df8 para obtener la columna con el numero de vuelos diarios por franja horaria en el aeropuerto de origen. 
# Creacion de la COLUMNA ADICIONAL 4
joined3=df7.join(df8, "DATE_HOUR_ORIGIN", 'inner')
display(joined3)

# COMMAND ----------

# COLUMNA ADICIONAL 5
# (1) Creamos tabla temporal para poder hacer la concatenacion posterior de la fecha, franja horaria y el aeropuerto de destino
joined3.registerTempTable("joined3")
# (1) Creamos el codigo unico de cada fila mediante la concatenacion de la fecha, franja horaria y el aeropuerto de destino
df9=sqlContext.sql("SELECT *, CONCAT(FL_DATE,',',DEST_AIRPORT_ID,',',ARR_TIME_BLK) as DATE_HOUR_DEST FROM joined3")
# (2) Creamos la tabla temporal para poder hacer la consulta y obtener la vista deseada
df9.registerTempTable("df9")
# (2) Realizamos la consulta para obtener el numero de vuelos al dia y por franja horaria en el aeropuerto de destino
df10=sqlContext.sql("SELECT DATE_HOUR_DEST, COUNT(DATE_HOUR_DEST) as N_FLIGHTS_HOUR_DEST FROM df9 GROUP BY DATE_HOUR_DEST")
# (3) Unimos los dataframes df9 y df10 para obtener la columna con el numero de vuelos diarios por franja horaria en el aeropuerto de destino. 
# Creacion de la COLUMNA ADICIONAL 5
joined4=df9.join(df10, "DATE_HOUR_DEST", 'inner')
display(joined4)

# COMMAND ----------

joined4.registerTempTable("joined4")

# COMMAND ----------

# Nos creamos la columna de numero de vuelo y fecha como concatenacion de las dos variables para poder construir posteriormente la columna adicional 8
df11 = sqlContext.sql("SELECT *, CONCAT(FL_DATE,',',FL_NUM) as DATE_FNUMBER FROM joined4")
display(df11)

# COMMAND ----------

# COLUMNA ADICIONAL 6: orden del vuelo en el aeropuerto de origen segun la hora de su despegue programado en el plan de vuelo (CRS_DEP_TIME)
df11.registerTempTable("df11")

# COMMAND ----------

df12=sqlContext.sql("SELECT * FROM df11 ORDER BY DATE_ORIGIN ASC, CRS_DEP_TIME ASC")
display(df12)

# COMMAND ----------

# Guardar df12 y leer con pandas
df_int = df12.select(trim(col("DATE_ORIGIN")).cast(StringType()).alias("DATE_ORIGIN"),
                trim(col("DATE_DEST")).cast(StringType()).alias("DATE_DEST"),
                trim(col("DATE_FNUMBER")).cast(StringType()).alias("DATE_FNUMBER"),
                trim(col("MONTH")).cast(IntegerType()).alias("MONTH"),
                trim(col("HOLIDAYS")).cast(IntegerType()).alias("HOLIDAYS"),
                trim(col("DAY_OF_MONTH")).cast(IntegerType()).alias("DAY_OF_MONTH"),
                trim(col("FL_DATE")).cast(StringType()).alias("FL_DATE"),
                trim(col("DAY_OF_WEEK")).cast(IntegerType()).alias("DAY_OF_WEEK"),
                trim(col("UNIQUE_CARRIER")).cast(StringType()).alias("UNIQUE_CARRIER"),
                trim(col("TAIL_NUM")).cast(StringType()).alias("TAIL_NUM"),
                trim(col("FL_NUM")).cast(IntegerType()).alias("FL_NUM"),
                trim(col("ORIGIN_AIRPORT_ID")).cast(IntegerType()).alias("ORIGIN_AIRPORT_ID"),
                trim(col("ORIGIN_CITY_MARKET_ID")).cast(IntegerType()).alias("ORIGIN_CITY_MARKET_ID"),
                trim(col("ORIGIN_STATE_NM")).cast(StringType()).alias("ORIGIN_STATE_NM"),
                trim(col("DEST_AIRPORT_ID")).cast(IntegerType()).alias("DEST_AIRPORT_ID"),
                trim(col("DEST_CITY_MARKET_ID")).cast(IntegerType()).alias("DEST_CITY_MARKET_ID"),
                trim(col("DEST_STATE_NM")).cast(StringType()).alias("DEST_STATE_ABR"),
                trim(col("CRS_DEP_TIME")).cast(IntegerType()).alias("CRS_DEP_TIME"),
                trim(col("DEP_TIME")).cast(IntegerType()).alias("DEP_TIME"),
                trim(col("DEP_DELAY")).cast(IntegerType()).alias("DEP_DELAY"),
                trim(col("DEP_DELAY_NEW")).cast(IntegerType()).alias("DEP_DELAY_NEW"),
                trim(col("DEP_DEL15")).cast(IntegerType()).alias("DEP_DEL15"),
                trim(col("DEP_DELAY_GROUP")).cast(IntegerType()).alias("DEP_DELAY_GROUP"),
                trim(col("DEP_TIME_BLK")).cast(StringType()).alias("DEP_TIME_BLK"),
                trim(col("TAXI_OUT")).cast(IntegerType()).alias("TAXI_OUT"),
                trim(col("WHEELS_OFF")).cast(IntegerType()).alias("WHEELS_OFF"),
                trim(col("TAXI_IN")).cast(IntegerType()).alias("TAXI_IN"),
                trim(col("CRS_ARR_TIME")).cast(IntegerType()).alias("CRS_ARR_TIME"),
                trim(col("ARR_TIME")).cast(IntegerType()).alias("ARR_TIME"),
                trim(col("ARR_DELAY")).cast(IntegerType()).alias("ARR_DELAY"),
                trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
                trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"),
                trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
                trim(col("ARR_TIME_BLK")).cast(StringType()).alias("ARR_TIME_BLK"),
                trim(col("CRS_ELAPSED_TIME")).cast(IntegerType()).alias("CRS_ELAPSED_TIME"),
                trim(col("ACTUAL_ELAPSED_TIME")).cast(IntegerType()).alias("ACTUAL_ELAPSED_TIME"),
                trim(col("DISTANCE")).cast(IntegerType()).alias("DISTANCE"),
                trim(col("DISTANCE_GROUP")).cast(IntegerType()).alias("DISTANCE_GROUP"),
                trim(col("WEATHER_DELAY")).cast(IntegerType()).alias("WEATHER_DELAY"),
                trim(col("N_FLIGHTS_DAY_ORIGIN")).cast(IntegerType()).alias("N_FLIGHTS_DAY_ORIGIN"),
                trim(col("N_FLIGHTS_DAY_DEST")).cast(IntegerType()).alias("N_FLIGHTS_DAY_DEST"),
                trim(col("N_FLIGHTS_HOUR_ORIGIN")).cast(IntegerType()).alias("N_FLIGHTS_HOUR_ORIGIN"),
                trim(col("N_FLIGHTS_HOUR_DEST")).cast(IntegerType()).alias("N_FLIGHTS_HOUR_DEST")
               )
display(df_int)

# COMMAND ----------

df_int.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/dataset/df_int")

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/dataset/df_int"))

# COMMAND ----------

# Copiamos el archivo para tenerlo en local y poder leerlo con pandas
dbutils.fs.cp("dbfs:/dataset/df_int/part-00000-3e020313-514a-4e98-a4c4-b38edad0a053-c000.csv", "file:/home/ubuntu/databricks/common/df_int_pd.csv", recurse=False)
display(dbutils.fs.ls("file:/home/ubuntu/databricks/common/df_int_pd.csv"))

# COMMAND ----------

import pandas as pd
df_int_pd = pd.read_csv("file:/home/ubuntu/databricks/common/df_int_pd.csv", sep=',', header='infer')
print(df_int_pd)

# COMMAND ----------

# COLUMNA ADICIONAL 7: Orden de salida de los vuelos (es decir, en el aeropuerto de origen)  
# Vuelos previamente ordenados por fecha-codigo_aeropuerto (DATE_ORIGIN) y hora de salida (CRS_DEP_TIME)
# Suponemos que a partir de las 00.00 empieza la actividad del aeropuerto, por lo que cada dia se pone a 0 el contador de salidas

order = []
number= []
row_0=[]
# row_0 = valor de la fila anterior a la que entra en el bucle (row)
# Valor inicial de la fila anterior (row_0) = "", ya que no esta en el dataframe

# Para cada row en la columna DATE_ORIGIN,

for row in df_int_pd['DATE_ORIGIN']:

    # Si el codigo del aeropuerto es igual al codigo del aeropuerto anterior (de row_0),
    if row == row_0:
      number = number + 1 
      # Anexa el numero como posicion de salida del vuelo en el aeropuerto 
      order.append(number)
    else: 
      number=1
      # Anexa el numero como posicion de salida del vuelo en el aeropuerto
      order.append(number)
      
    row_0=row
      
df_int_pd['ORDER_FLIGHT_ORIGIN'] = order
print(df_int_pd)

# COMMAND ----------

# COLUMNA ADICIONAL 8: Retraso acumulado en salida (Aeropuerto de origen)
# Vuelos previamente ordenados por fecha-codigo_aeropuerto (DATE_ORIGIN) y hora de salida (CRS_DEP_TIME) 
# Suponemos, al igual que en el caso anterior, que a partir de las 00.00 (con el inicio de un nuevo dia), se pone a 0 el contador

result_delay = []
row_0 = 0
delay_0 = 0

for row in df_int_pd["DATE_ORIGIN"].index:

  if df_int_pd["DATE_ORIGIN"][row] == df_int_pd["DATE_ORIGIN"][row_0]:
    delay = df_int_pd["DEP_DELAY_NEW"][row]
    acum = delay + delay_0
    result_delay.append(acum)
        
  else:
    delay = df_int_pd["DEP_DELAY_NEW"][row]
    acum = delay 
    result_delay.append(acum)
    
  delay_0 = acum
  row_0 = row
  
df_int_pd['DEP_DELAY_CUM'] = result_delay
print(df_int_pd)

# COMMAND ----------

# COLUMNAS ADICIONALES 9 y 10: Orden de llegada de los vuelos y retraso acumulado en llegada
# Ordenamos por fecha-aeropuerto_destino (DATE_DEST) y hora (CRS_ARR_TIME)

df_int_pd2 = df_int_pd.sort_values(by=["DATE_DEST", "CRS_ARR_TIME"], ascending=True)
print(df_int_pd2)  

# COMMAND ----------

# COLUMNA ADICIONAL 9: Orden de llegada de los vuelos (es decir, en el aeropuerto de origen)  

order = []
number= []
row_0=[]
# row_0 = valor de la fila anterior a la que entra en el bucle (row)
# Valor inicial de la fila anterior (row_0) = "", ya que no esta en el dataframe

# Para cada row en la columna DATE_DEST,

for row in df_int_pd2['DATE_DEST']:

    # Si el codigo del aeropuerto es igual al codigo del aeropuerto anterior (de row_0),
    if row == row_0:
      number = number + 1 
      # Anexa el numero como posicion de llegada del vuelo en el aeropuerto 
      order.append(number)
    else: 
      number=1
      # Anexa el numero como posicion de llegada del vuelo en el aeropuerto
      order.append(number)
      
    row_0=row
      
df_int_pd2['ORDER_FLIGHT_DEST'] = order
print(df_int_pd2)

# COMMAND ----------

# COLUMNA ADICIONAL 10: Retraso acumulado en llegadas 
# Vuelos previamente ordenados por fecha-codigo_aeropuerto (DATE_DEST) y hora de salida (CRS_ARR_TIME) 
# Suponemos, al igual que en el caso anterior, que a partir de las 00.00 (con el inicio de un nuevo dia), se pone a 0 el contador

result_delay = []
row_0 = 0
delay_0 = 0

for row in df_int_pd2["DATE_DEST"].index:

  if df_int_pd2["DATE_DEST"][row] == df_int_pd2["DATE_DEST"][row_0]:
    delay = df_int_pd2["ARR_DELAY_NEW"][row]
    acum = delay + delay_0
    result_delay.append(acum)
        
  else:
    delay = df_int_pd2["ARR_DELAY_NEW"][row]
    acum = delay 
    result_delay.append(acum)
    
  delay_0 = acum
  row_0 = row
  
df_int_pd2['ARR_DELAY_CUM'] = result_delay
print(df_int_pd2)

# COMMAND ----------

# Retraso anterior del numero de vuelo: ordernar por date_Fnumber y hora asc
df_int_pd3 = df_int_pd2.sort_values(by=["DATE_FNUMBER", "CRS_ARR_TIME"], ascending=True)
print(df_int_pd3)  

# COMMAND ----------

# COLUMNA ADICIONAL 11: retraso en llegadas del vuelo anterior con el mismo flight_number 
# Retraso anterior de vuelos con el mismo flight number (0-no retraso anterior, 1-retraso del vuelo anterior)

result_delay = []
row_0 = 0

for row in df_int_pd3["DATE_FNUMBER"].index:

  if df_int_pd3["DATE_FNUMBER"][row] == df_int_pd3["DATE_FNUMBER"][row_0]:
    delay_0 = df_int_pd3["ARR_DEL15"][row_0]
    if row == 0:
      delay_0 = 0
      result_delay.append(delay_0)
    else:
      result_delay.append(delay_0)
  else:
    delay_0 = 0 
    result_delay.append(delay_0)
    
  row_0 = row
  
df_int_pd3['ARR_DELAY_FNUMBER_BEFORE'] = result_delay
print(df_int_pd3)

# COMMAND ----------

# COLUMNA ADICIONAL 12: Retraso acumulado de llegada para ese numero de vuelo
# Vuelos previamente ordenados por fecha-flight_number (DATE_FNUMBER) y hora de llegada (CRS_ARR_TIME)

result_delay = []
row_0 = 0
delay_0 = 0

for row in df_int_pd3["DATE_FNUMBER"].index:

  if df_int_pd3["DATE_FNUMBER"][row] == df_int_pd3["DATE_FNUMBER"][row_0]:
    delay = df_int_pd3["ARR_DELAY_NEW"][row]
    acum = delay + delay_0 
    result_delay.append(acum)
        
  else:
    delay = df_int_pd3["ARR_DELAY_NEW"][row]
    acum = delay 
    result_delay.append(acum)
    
  delay_0 = acum
  row_0 = row
  
df_int_pd3['CUM_ARR_FNUMBER_DELAY'] = result_delay
print(df_int_pd3)

# COMMAND ----------

type(df_int_pd3)

# COMMAND ----------

#Para poder abrir el dataframe de pandas en sql lo tenemos que guardar primero en local como csv para leerlo posteriormente como sql
df_int_pd3.to_csv("/home/df_int_pd3.csv", sep=',')

# COMMAND ----------

display(dbutils.fs.ls("file:/home/df_int_pd3.csv"))

# COMMAND ----------

# Leemos el csv que habiamos guardado en local como sql 
df_int_pd3 = sqlContext.read.format("csv").option("header", "true").load("file:/home/df_int_pd3.csv")

# COMMAND ----------

# Nos quedamos con las columnas que nos interesaban

from pyspark.sql.types import *
from pyspark.sql.functions import trim, col

df_int_pd4 = df_int_pd3.select(trim(col("DATE_ORIGIN")).cast(StringType()).alias("DATE_ORIGIN"),
                trim(col("DATE_DEST")).cast(StringType()).alias("DATE_DEST"),
                trim(col("DATE_FNUMBER")).cast(StringType()).alias("DATE_FNUMBER"),
                trim(col("MONTH")).cast(IntegerType()).alias("MONTH"),
                trim(col("HOLIDAYS")).cast(IntegerType()).alias("HOLIDAYS"),
                trim(col("DAY_OF_MONTH")).cast(IntegerType()).alias("DAY_OF_MONTH"),
                trim(col("FL_DATE")).cast(StringType()).alias("FL_DATE"),
                trim(col("DAY_OF_WEEK")).cast(IntegerType()).alias("DAY_OF_WEEK"),
                trim(col("UNIQUE_CARRIER")).cast(StringType()).alias("UNIQUE_CARRIER"),
                trim(col("TAIL_NUM")).cast(StringType()).alias("TAIL_NUM"),
                trim(col("FL_NUM")).cast(IntegerType()).alias("FL_NUM"),
                trim(col("ORIGIN_AIRPORT_ID")).cast(IntegerType()).alias("ORIGIN_AIRPORT_ID"),
                trim(col("ORIGIN_CITY_MARKET_ID")).cast(IntegerType()).alias("ORIGIN_CITY_MARKET_ID"),
                trim(col("ORIGIN_STATE_NM")).cast(StringType()).alias("ORIGIN_STATE_NM"),
                trim(col("DEST_AIRPORT_ID")).cast(IntegerType()).alias("DEST_AIRPORT_ID"),
                trim(col("DEST_CITY_MARKET_ID")).cast(IntegerType()).alias("DEST_CITY_MARKET_ID"),
                trim(col("DEST_STATE_ABR")).cast(StringType()).alias("DEST_STATE_NM"),
                trim(col("CRS_DEP_TIME")).cast(IntegerType()).alias("CRS_DEP_TIME"),
                trim(col("DEP_TIME")).cast(IntegerType()).alias("DEP_TIME"),
                trim(col("DEP_DELAY")).cast(IntegerType()).alias("DEP_DELAY"),
                trim(col("DEP_DELAY_NEW")).cast(IntegerType()).alias("DEP_DELAY_NEW"),
                trim(col("DEP_DEL15")).cast(IntegerType()).alias("DEP_DEL15"),
                trim(col("DEP_DELAY_GROUP")).cast(IntegerType()).alias("DEP_DELAY_GROUP"),
                trim(col("DEP_TIME_BLK")).cast(StringType()).alias("DEP_TIME_BLK"),
                trim(col("TAXI_OUT")).cast(IntegerType()).alias("TAXI_OUT"),
                trim(col("WHEELS_OFF")).cast(IntegerType()).alias("WHEELS_OFF"),
                trim(col("TAXI_IN")).cast(IntegerType()).alias("TAXI_IN"),
                trim(col("CRS_ARR_TIME")).cast(IntegerType()).alias("CRS_ARR_TIME"),
                trim(col("ARR_TIME")).cast(IntegerType()).alias("ARR_TIME"),
                trim(col("ARR_DELAY")).cast(IntegerType()).alias("ARR_DELAY"),
                trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
                trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"),
                trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
                trim(col("ARR_TIME_BLK")).cast(StringType()).alias("ARR_TIME_BLK"),
                trim(col("CRS_ELAPSED_TIME")).cast(IntegerType()).alias("CRS_ELAPSED_TIME"),
                trim(col("ACTUAL_ELAPSED_TIME")).cast(IntegerType()).alias("ACTUAL_ELAPSED_TIME"),
                trim(col("DISTANCE")).cast(IntegerType()).alias("DISTANCE"),
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
                trim(col("CUM_ARR_FNUMBER_DELAY")).cast(IntegerType()).alias("CUM_ARR_FNUMBER_DELAY")
               )
display(df_int_pd4)

# COMMAND ----------

df_int_pd4.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/dataset/df_int_pd4")

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/dataset/df_int_pd4"))

# COMMAND ----------

# Copiamos el archivo para tenerlo en local y poder leerlo con pandas
dbutils.fs.cp("dbfs:/dataset/df_int_pd4/part-00000-7df41518-5555-4ec9-923d-aa93734ef471-c000.csv", "file:/home/ubuntu/databricks/common/df_int_pd4.csv", recurse=False)
display(dbutils.fs.ls("file:/home/ubuntu/databricks/common/df_int_pd4.csv"))

# COMMAND ----------

import pandas as pd
df_int_pd4 = pd.read_csv("file:/home/ubuntu/databricks/common/df_int_pd4.csv", sep=',', header='infer')
# print(df_int_pd4)

# COMMAND ----------

# Continuamos con la creacion de las columnas adicionales. Para ello necesitamos ordenar primero por date_Fnumber y hora asc
df_int_pd5 = df_int_pd4.sort_values(by=["DATE_FNUMBER", "CRS_DEP_TIME"], ascending=True)
print(df_int_pd5)  

# COMMAND ----------

# COLUMNA ADICIONAL 13: retraso en salidas del vuelo anterior con el mismo flight_number 
# Retraso anterior de vuelos con el mismo flight number (0-no retraso anterior, 1-retraso del vuelo anterior)

result_delay = []
row_0 = 0

for row in df_int_pd4["DATE_FNUMBER"].index:

  if df_int_pd5["DATE_FNUMBER"][row] == df_int_pd5["DATE_FNUMBER"][row_0]:
    delay_0 = df_int_pd5["DEP_DEL15"][row_0]
    if row == 0:
      delay_0 = 0
      result_delay.append(delay_0)
    else:
      result_delay.append(delay_0)
  else:
    delay_0 = 0 
    result_delay.append(delay_0)
    
  row_0 = row
  
df_int_pd5['DEP_DELAY_FNUMBER_BEFORE'] = result_delay
print(df_int_pd5)

# COMMAND ----------

# COLUMNA ADICIONAL 14: Retraso acumulado de salida para ese numero de vuelo
# Vuelos previamente ordenados por fecha-flight_number (DATE_FNUMBER) y hora de salida (CRS_DEP_TIME)

result_delay = []
row_0 = 0
delay_0 = 0

for row in df_int_pd5["DATE_FNUMBER"].index:

  if df_int_pd5["DATE_FNUMBER"][row] == df_int_pd5["DATE_FNUMBER"][row_0]:
    delay = df_int_pd5["DEP_DELAY_NEW"][row]
    acum = delay + delay_0 
    result_delay.append(acum)
        
  else:
    delay = df_int_pd5["DEP_DELAY_NEW"][row]
    acum = delay 
    result_delay.append(acum)
    
  delay_0 = acum
  row_0 = row
  
df_int_pd5['CUM_DEP_FNUMBER_DELAY'] = result_delay
print(df_int_pd5)

# COMMAND ----------

df_int_pd5.to_csv("/home/df_int_pd5.csv", sep=',')

# COMMAND ----------

# Leemos el csv que habiamos guardado en local como sql 
df_int_pd5 = sqlContext.read.format("csv").option("header", "true").load("file:/home/df_int_pd5.csv")
display(df_int_pd5)

# COMMAND ----------

# StringIndexer -> Convertimos las variables de tipo string en variables numericas
# UNIQUE_CARRIER, TAIL_NUM, ORIGIN_STATE_NM, DEST_STATE_NM, DEP_TIME_BLK, ARR_TIME_BLK

from pyspark.ml.feature import StringIndexer

indexer_1 = StringIndexer(inputCol="UNIQUE_CARRIER", outputCol="UNIQUE_CARRIER_Index")
indexed_1 = indexer_1.fit(df_int_pd5).transform(df_int_pd5)

indexer_2 = StringIndexer(inputCol="TAIL_NUM", outputCol="TAIL_NUM_Index")
indexed_2 = indexer_2.fit(indexed_1).transform(indexed_1)

indexer_3 = StringIndexer(inputCol="ORIGIN_STATE_NM", outputCol="ORIGIN_STATE_NM_Index")
indexed_3 = indexer_3.fit(indexed_2).transform(indexed_2)

display(indexed_3)

# COMMAND ----------

indexed_3.registerTempTable("indexed_3")

# COMMAND ----------

# MAGIC %sql select distinct DEP_TIME_BLK from indexed_3 order by DEP_TIME_BLK asc

# COMMAND ----------

from pyspark.sql.types import *

def categorize_time_blk (s) :
  if s == "0001-0559": return 0;
  elif s =="0600-0659": return 1;
  elif s =="0700-0759": return 2;
  elif s =="0800-0859": return 3;
  elif s =="0900-0959": return 4;
  elif s =="1000-1059": return 5;
  elif s =="1100-1159": return 6;
  elif s =="1200-1259": return 7;
  elif s =="1300-1359": return 8;
  elif s =="1400-1459": return 9;
  elif s =="1500-1559": return 10;
  elif s =="1600-1659": return 11;
  elif s =="1700-1759": return 12;
  elif s =="1800-1859": return 13;
  elif s =="1900-1959": return 14;
  elif s =="2000-2059": return 15;
  elif s =="2100-2159": return 16;
  elif s =="2200-2259": return 17;
  elif s =="2300-2359": return 18;
  
udf_categorize_time_blk  = udf(categorize_time_blk, IntegerType())
indexed_4 = indexed_3.withColumn("DEP_TIME_BLK_Index", udf_categorize_time_blk("DEP_TIME_BLK"))
indexed_5 = indexed_4.withColumn("ARR_TIME_BLK_Index", udf_categorize_time_blk("ARR_TIME_BLK"))
display(indexed_5)

# COMMAND ----------

indexed_5.registerTempTable("indexed_5")

# COMMAND ----------

display(indexed_5)

# COMMAND ----------

indexed_6 = sqlContext.sql("SELECT ORIGIN_STATE_NM_Index as DEST_STATE_NM_Index, ORIGIN_STATE_NM as DEST_STATE_NM FROM indexed_5 GROUP BY ORIGIN_STATE_NM, ORIGIN_STATE_NM_Index ORDER BY ORIGIN_STATE_NM_Index asc")
display(indexed_6)

# COMMAND ----------

joined_indexed=indexed_5.join(indexed_6, "DEST_STATE_NM", 'inner')
display(joined_indexed)

# COMMAND ----------

display(joined_indexed)

# COMMAND ----------

joined_indexed.registerTempTable("joined_indexed")

# COMMAND ----------

# MAGIC %sql select distinct DEST_STATE_NM_Index from joined_indexed group by DEST_STATE_NM_Index order by DEST_STATE_NM_Index asc

# COMMAND ----------

#CAMBIAR DEP_DELAY_GROUP Y ARR_DELAY_GROUP (-1,-2 por 13,14)

def categorize_delay_group (s) :
  if s == "-1": return 13;
  elif s == "-2": return 14;
  elif s == "0": return 0;
  elif s == "1": return 1;
  elif s == "2": return 2;
  elif s == "3": return 3;
  elif s == "4": return 4;
  elif s == "5": return 5;
  elif s == "6": return 6;
  elif s == "7": return 7;
  elif s == "8": return 8;
  elif s == "9": return 9;
  elif s == "10": return 10;
  elif s == "11": return 11;
  elif s == "12": return 12;
  
udf_categorize_delay_group  = udf(categorize_delay_group, IntegerType())
joined_indexed1 = joined_indexed.withColumn("DEP_DELAY_GROUP_Index", udf_categorize_delay_group("DEP_DELAY_GROUP"))
joined_indexed2 = joined_indexed1.withColumn("ARR_DELAY_GROUP_Index", udf_categorize_delay_group("ARR_DELAY_GROUP"))
display(joined_indexed2)

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import trim, col

joined_indexed3 = joined_indexed2.select(trim(col("MONTH")).cast(IntegerType()).alias("MONTH"),
                trim(col("HOLIDAYS")).cast(IntegerType()).alias("HOLIDAYS"),
                trim(col("DAY_OF_MONTH")).cast(IntegerType()).alias("DAY_OF_MONTH"),
                trim(col("DAY_OF_WEEK")).cast(IntegerType()).alias("DAY_OF_WEEK"),
                trim(col("UNIQUE_CARRIER_Index")).cast(IntegerType()).alias("UNIQUE_CARRIER"),
                trim(col("TAIL_NUM_Index")).cast(IntegerType()).alias("TAIL_NUM"),
                trim(col("FL_NUM")).cast(IntegerType()).alias("FL_NUM"),
                trim(col("ORIGIN_AIRPORT_ID")).cast(IntegerType()).alias("ORIGIN_AIRPORT_ID"),
                trim(col("ORIGIN_CITY_MARKET_ID")).cast(IntegerType()).alias("ORIGIN_CITY_MARKET_ID"),
                trim(col("ORIGIN_STATE_NM_Index")).cast(IntegerType()).alias("ORIGIN_STATE_NM"),
                trim(col("DEST_AIRPORT_ID")).cast(IntegerType()).alias("DEST_AIRPORT_ID"),
                trim(col("DEST_CITY_MARKET_ID")).cast(IntegerType()).alias("DEST_CITY_MARKET_ID"),
                trim(col("DEST_STATE_NM_Index")).cast(IntegerType()).alias("DEST_STATE_NM"),
                trim(col("CRS_DEP_TIME")).cast(IntegerType()).alias("CRS_DEP_TIME"),
                trim(col("DEP_TIME")).cast(IntegerType()).alias("DEP_TIME"),
                trim(col("DEP_DELAY")).cast(IntegerType()).alias("DEP_DELAY"),
                trim(col("DEP_DELAY_NEW")).cast(IntegerType()).alias("DEP_DELAY_NEW"),
                trim(col("DEP_DEL15")).cast(IntegerType()).alias("DEP_DEL15"),
                trim(col("DEP_DELAY_GROUP_Index")).cast(IntegerType()).alias("DEP_DELAY_GROUP"),
                trim(col("DEP_TIME_BLK_Index")).cast(IntegerType()).alias("DEP_TIME_BLK"),
                trim(col("TAXI_OUT")).cast(IntegerType()).alias("TAXI_OUT"),
                trim(col("WHEELS_OFF")).cast(IntegerType()).alias("WHEELS_OFF"),
                trim(col("TAXI_IN")).cast(IntegerType()).alias("TAXI_IN"),
                trim(col("CRS_ARR_TIME")).cast(IntegerType()).alias("CRS_ARR_TIME"),
                trim(col("ARR_TIME")).cast(IntegerType()).alias("ARR_TIME"),
                trim(col("ARR_DELAY")).cast(IntegerType()).alias("ARR_DELAY"),
                trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
                trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"),
                trim(col("ARR_DELAY_GROUP_Index")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
                trim(col("ARR_TIME_BLK_Index")).cast(IntegerType()).alias("ARR_TIME_BLK"),
                trim(col("CRS_ELAPSED_TIME")).cast(IntegerType()).alias("CRS_ELAPSED_TIME"),
                trim(col("ACTUAL_ELAPSED_TIME")).cast(IntegerType()).alias("ACTUAL_ELAPSED_TIME"),
                trim(col("DISTANCE")).cast(IntegerType()).alias("DISTANCE"),
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
display(joined_indexed3)

# COMMAND ----------

joined_indexed3.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/dataset/df_preprocesado")

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/dataset/df_preprocesado"))

# COMMAND ----------

# Realizamos la ultima transformacion que nos habiamos dejado anteriormente atras. Como los tiempos no estan en horas sino en numeros, hay que pasar los minutos de los tiempos de vuelo a numeros.
# Siendo la diferencia entre la llegada y la salida.
# Para ello copiamos en local el dataframe que habiamos creado como definitivo para que nos pueda caber el que creemos a continuacion 

dbutils.fs.cp("dbfs:/dataset/df_preprocesado/part-00000-193fb8ab-b1fa-4973-8fd0-86132267e526-c000.csv", "file:/home/ubuntu/databricks/common/df_preprocesado.csv", recurse=False)
display(dbutils.fs.ls("file:/home/ubuntu/databricks/common/df_preprocesado.csv"))

# COMMAND ----------

# Leemos el archivo y lo declaramos como la variable df_procesado
df_preprocesado = sqlContext.read.format("csv").option("header", "true").load("file:/home/ubuntu/databricks/common/df_preprocesado.csv")

# COMMAND ----------

df_preprocesado.registerTempTable("df_preprocesado")

# COMMAND ----------

# Restamos por ultimo las columnas CRS_ARR_TIME y CRS_DEP_TIME para obtener el tiempo previsto de vuelo segun su plan de vuelo (CRS_ELAPSED TIME). 
# Y realizamos la misma operacion para obtener el tiempo real de este.
df_resta = sqlContext.sql("select *, (CRS_ARR_TIME-CRS_DEP_TIME) as CRS_ELAPSED_TIME_numero, (ARR_TIME-DEP_TIME) as ACTUAL_ELAPSED_TIME_numero from df_preprocesado")
display(df_resta)

# COMMAND ----------

from pyspark.sql.types import *
from pyspark.sql.functions import trim, col

df_resta2 = df_resta.select(trim(col("MONTH")).cast(IntegerType()).alias("MONTH"),
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
                trim(col("TAXI_IN")).cast(IntegerType()).alias("TAXI_IN"),
                trim(col("CRS_ARR_TIME")).cast(IntegerType()).alias("CRS_ARR_TIME"),
                trim(col("ARR_TIME")).cast(IntegerType()).alias("ARR_TIME"),
                trim(col("ARR_DELAY")).cast(IntegerType()).alias("ARR_DELAY"),
                trim(col("ARR_DELAY_NEW")).cast(IntegerType()).alias("ARR_DELAY_NEW"),
                trim(col("ARR_DEL15")).cast(IntegerType()).alias("ARR_DEL15"),
                trim(col("ARR_DELAY_GROUP")).cast(IntegerType()).alias("ARR_DELAY_GROUP"),
                trim(col("ARR_TIME_BLK")).cast(IntegerType()).alias("ARR_TIME_BLK"),
                trim(col("CRS_ELAPSED_TIME_numero")).cast(IntegerType()).alias("CRS_ELAPSED_TIME_numero"),
                trim(col("ACTUAL_ELAPSED_TIME_numero")).cast(IntegerType()).alias("ACTUAL_ELAPSED_TIME_numero"),
                trim(col("DISTANCE")).cast(IntegerType()).alias("DISTANCE"),
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
display(df_resta2)

# COMMAND ----------

# Guardamos el dataframe definitivo con todas las columnas transformadas
df_resta2.coalesce(1).write.format("com.databricks.spark.csv").option("header", "true").save("dbfs:/dataset/df_preprocesado")

# COMMAND ----------

display(dbutils.fs.ls("dbfs:/dataset/df_preprocesado"))

# COMMAND ----------

# FIN DEL PREPROCESAMIENTO DE LOS DATOS
