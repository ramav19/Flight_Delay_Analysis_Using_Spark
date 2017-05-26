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

newData.createOrReplaceTempView("FlightData")

// COMMAND ----------

//Primary Cause of Flight Delay
val causeD = sqlContext.sql("SELECT sum(WEATHER_DELAY) Weather,sum(NAS_DELAY) NAS,sum(SECURITY_DELAY) Security,sum(LATE_AIRCRAFT_DELAY) lateAircraft,sum(CARRIER_DELAY) Carrier FROM FlightData ")
causeD.show()

// COMMAND ----------

//Which airports have most delay
//val groupDelay = sqlContext.sql("SELECT ORIGIN_CITY_NAME , count(*) FlightDelay FROM FlightData WHERE DEP_DELAY > 15 GROUP BY ORIGIN_CITY_NAME ORDER BY FlightCount desc" )
//val groupTotal = sqlContext.sql("SELECT ORIGIN_CITY_NAME , count(*) FlightTotal FROM FlightData GROUP BY ORIGIN_CITY_NAME ORDER BY FlightCount desc" )

//val groupFinal = sqlContext.sql("select * from groupDelay, groupTotal where groupDelay.ORIGIN_CITY_NAME = groupTotal.ORIGIN_CITY_NAME")

val airportFinal = sqlContext.sql("select t1.ORIGIN_CITY_NAME, t1.FlightDelay, coalesce(t2.FlightTotal, 0) as FlightTotal, (t1.FlightDelay * 100 /t2.FlightTotal) as Percentage from (SELECT ORIGIN_CITY_NAME , count(*) FlightDelay FROM FlightData WHERE DEP_DELAY > 15 GROUP BY ORIGIN_CITY_NAME) t1 left join (SELECT ORIGIN_CITY_NAME ,count(*) FlightTotal FROM FlightData GROUP BY ORIGIN_CITY_NAME) t2 on t1.ORIGIN_CITY_NAME = t2.ORIGIN_CITY_NAME ORDER BY t1.FlightDelay desc")

airportFinal.show()

// COMMAND ----------

//Carriers which have more delays 
val carrierD = sqlContext.sql("select t1.UNIQUE_CARRIER, t1.FlightDelay, coalesce(t2.FlightTotal, 0) as FlightTotal, (t1.FlightDelay * 100 /t2.FlightTotal) as Percentage from (SELECT UNIQUE_CARRIER , count(*) FlightDelay FROM FlightData WHERE DEP_DELAY > 15 GROUP BY UNIQUE_CARRIER) t1 left join (SELECT UNIQUE_CARRIER ,count(*) FlightTotal FROM FlightData GROUP BY UNIQUE_CARRIER) t2 on t1.UNIQUE_CARRIER = t2.UNIQUE_CARRIER ORDER BY t1.FlightDelay desc")

carrierD.show()

// COMMAND ----------

//Routes with the most delays
val routeD = sqlContext.sql("select t1.ORIGIN_CITY_NAME, t1.DEST_CITY_NAME, t1.FlightDelay, coalesce(t2.FlightTotal, 0) as FlightTotal, (t1.FlightDelay * 100 /t2.FlightTotal) as Percentage from (SELECT ORIGIN_CITY_NAME, DEST_CITY_NAME , count(*) FlightDelay FROM FlightData WHERE DEP_DELAY > 15 GROUP BY ORIGIN_CITY_NAME, DEST_CITY_NAME) t1 left join (SELECT ORIGIN_CITY_NAME, DEST_CITY_NAME ,count(*) FlightTotal FROM FlightData GROUP BY ORIGIN_CITY_NAME, DEST_CITY_NAME) t2 on t1.ORIGIN_CITY_NAME = t2.ORIGIN_CITY_NAME AND t1.DEST_CITY_NAME = t2.DEST_CITY_NAME ORDER BY t1.FlightDelay desc")

routeD.show()

// COMMAND ----------

//Delays on days of the week

val dayDelay = sqlContext.sql("SELECT t1.DAY_OF_WEEK, t1.FlightsDelayed, coalesce(t2.FlightsOnDay, 0) as FlightsOnDay, (t1.FlightsDelayed*100/t2.FlightsOnDay) AS Percentage FROM (SELECT DAY_OF_WEEK, Count(*) AS FlightsDelayed FROM FlightData WHERE DEP_DELAY > 15 GROUP BY DAY_OF_WEEK) t1 LEFT JOIN (SELECT DAY_OF_WEEK, Count(*) AS FlightsOnDay FROM FlightData GROUP BY DAY_OF_WEEK) t2 ON t1.DAY_OF_WEEK = t2.DAY_OF_WEEK ORDER BY DAY_OF_WEEK ASC")
dayDelay.show()

// COMMAND ----------

//Delays on months
val monthDelay = sqlContext.sql("SELECT t1.MONTH, t1.FlightsDelayed, coalesce(t2.FlightsInMonth, 0) as FlightsInMonth, (t1.FlightsDelayed*100/t2.FlightsInMonth) AS Percentage FROM (SELECT MONTH, Count(*) AS FlightsDelayed FROM FlightData WHERE DEP_DELAY > 15 GROUP BY MONTH) t1 LEFT JOIN (SELECT MONTH, Count(*) AS FlightsInMonth FROM FlightData WHERE GROUP BY MONTH) t2 ON t1.MONTH = t2.MONTH ORDER BY MONTH ASC")
monthDelay.show()

// COMMAND ----------


