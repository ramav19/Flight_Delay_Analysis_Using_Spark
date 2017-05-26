// Databricks notebook source

val data= sqlContext.read.format("csv").option("header","true").option("inferSchema","true").load("/FileStore/tables/vkk36kjx1493067563752/")


// COMMAND ----------

data.printSchema()

// COMMAND ----------

data.createOrReplaceTempView("FlightData")

import org.apache.spark.sql.functions._
val newData = data.withColumn("DEP_DELAY", when($"DEP_DELAY" < 0,0).otherwise($"DEP_DELAY"))
                  .withColumn("ARR_DELAY", when($"ARR_DELAY" < 0,0).otherwise($"ARR_DELAY"))
newData.show()



// COMMAND ----------

//Placing label in first column
val midData = newData.select("YEAR", "DEP_DEL15", "DEP_DELAY", "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "FL_DATE", "UNIQUE_CARRIER", "ORIGIN_AIRPORT_ID", "ORIGIN_CITY_MARKET_ID", "ORIGIN_CITY_NAME", "DEST_AIRPORT_ID", "DEST_CITY_NAME", "CRS_DEP_TIME", "DEP_DELAY_GROUP", "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY", "ARR_DELAY_GROUP", "DISTANCE", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY", "_c25")

//midData.printSchema()
//midData.show()

// COMMAND ----------

//Removing Unnecessary Columns 
val midData_Filtered = newData.select("YEAR" ,"DEP_DEL15", "DEP_DELAY", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "UNIQUE_CARRIER", "ORIGIN_AIRPORT_ID", "ORIGIN_CITY_MARKET_ID", "ORIGIN_CITY_NAME", "DEST_AIRPORT_ID", "DEST_CITY_NAME", "CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE")

//Removing NULL values
var train_na_removed = midData_Filtered.na.drop()


// COMMAND ----------

println(midData_Filtered.count())

// COMMAND ----------

println(train_na_removed.count())

// COMMAND ----------


//Running StringIndexer to get Data
import org.apache.spark.ml.feature.StringIndexer

val indexer1 = new StringIndexer()
  .setInputCol("UNIQUE_CARRIER")
  .setOutputCol("UNIQUE_CARRIER_INDEX")
val indexed1 = indexer1.fit(train_na_removed).transform(train_na_removed)

val indexer2 = new StringIndexer()
  .setInputCol("ORIGIN_CITY_NAME")
  .setOutputCol("ORIGIN_CITY_NAME_INDEX")
val indexed2 = indexer2.fit(indexed1).transform(indexed1)

val indexer3 = new StringIndexer()
  .setInputCol("DEST_CITY_NAME")
  .setOutputCol("DEST_CITY_NAME_INDEX")
val indexed3 = indexer3.fit(indexed2).transform(indexed2)

//indexed3.printSchema()
val midData_temp = indexed3.select("YEAR", "DEP_DEL15", "DEP_DELAY", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "UNIQUE_CARRIER_INDEX", "ORIGIN_AIRPORT_ID", "ORIGIN_CITY_MARKET_ID", "ORIGIN_CITY_NAME_INDEX", "DEST_AIRPORT_ID", "DEST_CITY_NAME_INDEX", "CRS_DEP_TIME", "CRS_ARR_TIME", "DISTANCE")

midData_temp.printSchema()
//indexed.show()

// COMMAND ----------

import org.apache.spark.ml.{Pipeline, PipelineModel}
//Applying OneHotEncoder
import org.apache.spark.ml.feature.OneHotEncoder

val encoder1 = new OneHotEncoder()
  .setInputCol("UNIQUE_CARRIER_INDEX")
  .setOutputCol("UNIQUE_CARRIER_VEC")
//val encoded1 = encoder1.transform(midData_temp)

val encoder2 = new OneHotEncoder()
  .setInputCol("ORIGIN_CITY_NAME_INDEX")
  .setOutputCol("ORIGIN_CITY_NAME_VEC")
//val encoded2 = encoder2.transform(encoded1)

val encoder3 = new OneHotEncoder()
  .setInputCol("DEST_CITY_NAME_INDEX")
  .setOutputCol("DEST_CITY_NAME_VEC")
//val encoded3 = encoder3.transform(encoded2)

val encoder4 = new OneHotEncoder()
  .setInputCol("MONTH")
  .setOutputCol("MONTH_VEC")
//val encoded4 = encoder4.transform(encoded3)

val encoder5 = new OneHotEncoder()
  .setInputCol("DAY_OF_MONTH")
  .setOutputCol("DAY_OF_MONTH_VEC")
//val encoded5 = encoder5.transform(encoded4)

val encoder6 = new OneHotEncoder()
  .setInputCol("DAY_OF_WEEK")
  .setOutputCol("DAY_OF_WEEK_VEC")
//val encoded6 = encoder6.transform(encoded5)

val encoder7 = new OneHotEncoder()
  .setInputCol("ORIGIN_AIRPORT_ID")
  .setOutputCol("ORIGIN_AIRPORT_ID_VEC")
//val encoded7 = encoder7.transform(encoded6)

val encoder8 = new OneHotEncoder()
  .setInputCol("ORIGIN_CITY_MARKET_ID")
  .setOutputCol("ORIGIN_CITY_MARKET_ID_VEC")
//val encoded8 = encoder8.transform(encoded7)

val encoder9 = new OneHotEncoder()
  .setInputCol("DEST_AIRPORT_ID")
  .setOutputCol("DEST_AIRPORT_ID_VEC")
//val encoded9 = encoder9.transform(encoded8)

val encoder10 = new OneHotEncoder()
  .setInputCol("CRS_DEP_TIME")
  .setOutputCol("CRS_DEP_TIME_VEC")
//val encoded10 = encoder10.transform(encoded9)

val encoder11 = new OneHotEncoder()
  .setInputCol("CRS_ARR_TIME")
  .setOutputCol("CRS_ARR_TIME_VEC")
//val encoded11 = encoder11.transform(encoded10)

//encoded3.printSchema()

val pipeline = new Pipeline()
  .setStages(Array(encoder1, encoder2, encoder3, encoder4, encoder5, encoder6, encoder7, encoder8, encoder9, encoder10, encoder11))

//Run the feature transformations.
val pipelineModel = pipeline.fit(midData_temp)
val midData_temp2 = pipelineModel.transform(midData_temp)

val midData_temp3 = midData_temp2.select("DEP_DEL15", "MONTH_VEC", "DAY_OF_MONTH_VEC", "DAY_OF_WEEK_VEC", "UNIQUE_CARRIER_VEC", "ORIGIN_AIRPORT_ID_VEC", "ORIGIN_CITY_MARKET_ID_VEC", "ORIGIN_CITY_NAME_VEC", "DEST_AIRPORT_ID_VEC", "DEST_CITY_NAME_VEC", "DISTANCE")

//midData_temp2.printSchema()
midData_temp3.show()

// COMMAND ----------

//import org.apache.spark.ml.feature.VectorAssembler
//import org.apache.spark.ml.linalg.Vectors

//val assembler = new VectorAssembler()  .setInputCols(Array("DEP_DEL15", "MONTH_VEC", "DAY_OF_MONTH_VEC", "DAY_OF_WEEK_VEC", "UNIQUE_CARRIER_VEC", "ORIGIN_AIRPORT_ID_VEC", "ORIGIN_CITY_MARKET_ID_VEC", "ORIGIN_CITY_NAME_VEC", "DEST_AIRPORT_ID_VEC", "DEST_CITY_NAME_VEC", "DISTANCE"))  .setOutputCol("features")

//val midData_temp4 = assembler.transform(midData_temp3)
//println("Assembled columns to vector column 'features'")

//val midData_temp5 = midData_temp4.select("DEP_DEL15", "MONTH_VEC", "DAY_OF_MONTH_VEC", "DAY_OF_WEEK_VEC", "UNIQUE_CARRIER_VEC", "ORIGIN_AIRPORT_ID_VEC", "ORIGIN_CITY_MARKET_ID_VEC", "ORIGIN_CITY_NAME_VEC", "DEST_AIRPORT_ID_VEC", "DEST_CITY_NAME_VEC", "features")
//val midData_temp6 = midData_temp5.toDF("label", "MONTH_VEC", "DAY_OF_MONTH_VEC", "DAY_OF_WEEK_VEC", "UNIQUE_CARRIER_VEC", "ORIGIN_AIRPORT_ID_VEC", "ORIGIN_CITY_MARKET_ID_VEC", "ORIGIN_CITY_NAME_VEC", "DEST_AIRPORT_ID_VEC", "DEST_CITY_NAME_VEC", "features")
//midData_temp6.printSchema()
//midData_temp3.show()

// COMMAND ----------

//Picking different features 

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors

val assembler1 = new VectorAssembler()
  .setInputCols(Array("YEAR" ,"MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "DISTANCE", "UNIQUE_CARRIER_VEC", "ORIGIN_AIRPORT_ID_VEC"))
  .setOutputCol("features")

val midData_temp4 = assembler1.transform(midData_temp2)
println("Assembled columns to vector column 'features'")

val midData_temp5 = midData_temp4.select("DEP_DEL15", "features")
//val midData_temp5 = midData_temp4.select("DEP_DEL15", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK_VEC", "UNIQUE_CARRIER_VEC", "ORIGIN_AIRPORT_ID_VEC", "ORIGIN_CITY_NAME_VEC", "features")

val midData_temp6 = midData_temp5.toDF("label", "features")
midData_temp6.printSchema()


// COMMAND ----------

val Array(trainingData, testData) = midData_temp6.randomSplit(Array(0.7, 0.3))


// COMMAND ----------

println(trainingData.count())
println(testData.count())

// COMMAND ----------

//Code for Logistic Regression

import org.apache.spark.ml.classification.LogisticRegression

// Create initial LogisticRegression model
val lr = new LogisticRegression()
  .setMaxIter(10)
  .setRegParam(0.3)
  .setElasticNetParam(0.8)

//Train model with Training Data
val lrModel = lr.fit(trainingData)
val predictions = lrModel.transform(testData)
predictions.printSchema()

// COMMAND ----------

//Evaluating Results from Logistic Regression

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("label")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println("Test Error = " + (1.0 - accuracy))

// COMMAND ----------

println(accuracy)
