# Databricks notebook source
# MAGIC %md
# MAGIC #### Big Data Management Tools Final Assignment: Predicting Youtube Channel Popularity
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Load the data file.

# COMMAND ----------


dataPath = "dbfs:/FileStore/shared_uploads/cameron.turner@sympatico.ca/youtube_channels_1M_modified-5.csv"

youtube_df = spark.read.format("csv").option("header", "true").option("mode", "PERMISSIVE").option("inferSchema", "true").load(dataPath)


# COMMAND ----------

display(youtube_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Remove Null Values

# COMMAND ----------

# Fill String field unknowns.

youtube_df = youtube_df.na.fill(value='Unknown', subset=["country"])
youtube_df = youtube_df.na.fill(value='Unknown', subset=["channel_name"])

# Fill Integer field unknowns.

youtube_df = youtube_df.fillna(0, subset=['subscriber_count', 'total_views', 'total_videos', 'mean_views_last_30_videos', 'median_views_last_30_videos', 'std_views_last_30_videos', 'videos_per_week'])


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Add a new column for a Classification target, <i>total_views_cat</i>, to run Classification ML models.

# COMMAND ----------

from pyspark.sql.functions import when

# Create a new column, total_views_cat, that contains String classifications of the total_views.

youtube_df = youtube_df.withColumn('total_views_cat', when(youtube_df.total_views > 1000000, 'Excellent').otherwise(
                                   when(youtube_df.total_views > 500000, 'Great').otherwise(
                                   when(youtube_df.total_views > 100000, 'Good').otherwise('Okay'))))


# COMMAND ----------

# Verify the dataframe.

display(youtube_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create a relational table from the dataframe to run SQL queries.

# COMMAND ----------

# Create a table on which SQL commands can be run.

youtube_df.createOrReplaceTempView("youtube_data")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Exploratory Data Analysis

# COMMAND ----------

# MAGIC %sql
# MAGIC select total_views_cat, count(*) from youtube_data group by total_views_cat

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from youtube_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from youtube_data where total_views is null

# COMMAND ----------

# MAGIC %sql
# MAGIC select country from youtube_data where country is null

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct(country) from youtube_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select country, count(*) from youtube_data group by country order by count(*) desc limit 50

# COMMAND ----------

# MAGIC %sql
# MAGIC select country, count(*) from youtube_data group by country order by count(*) desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select country, sum(total_views) from youtube_data group by country order by sum(total_views) desc

# COMMAND ----------

# MAGIC %sql
# MAGIC select min(total_views), max(total_views), (max(total_views) - min(total_views)), std(total_views), median(total_views), avg(total_views) from youtube_data

# COMMAND ----------

# MAGIC %sql
# MAGIC select total_views from youtube_data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create Integer Columns for String Columns, <i>country</i> and <i>total_views_cat</i>, as <i>country_indexed</i> and <i>tvcat_indexed</i>.

# COMMAND ----------

from pyspark.sql.functions import corr
from pyspark.ml.feature import StringIndexer

# Create Integer representations of the String columns, country and total_views_cat.

country_indexer = StringIndexer(inputCol="country", outputCol="country_indexed")
tvcat_indexer   = StringIndexer(inputCol="total_views_cat", outputCol="tvcat_indexed")
countryIndexerModel = country_indexer.fit(youtube_df)
tvcatIndexerModel   = tvcat_indexer.fit(youtube_df)

# Transform the DataFrame using the fitted StringIndexer model

indexed_df = countryIndexerModel.transform(youtube_df)
new_indexed_df = tvcatIndexerModel.transform(indexed_df)

# Reset the main dataframe, youtube_df.

youtube_df = new_indexed_df

# Verify that the Integer columns were created.

display(youtube_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Correlations based on categorized total views Integer column, <i>tvcat_indexed</i>.

# COMMAND ----------

from pyspark.sql.functions import corr

# See if the country Integer column and the subscriber_count Integer column have strong correlations with the total views
# categorized Integer target.

youtube_df.select(corr("country_indexed", "tvcat_indexed")).show()
youtube_df.select(corr("subscriber_count", "tvcat_indexed")).show()


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Integer correlations based on original (non-categorized) <i>total_views</i> column.

# COMMAND ----------

from pyspark.sql.functions import corr

# See if the Integer columns have strong correlations with the total views Integer target.

youtube_df.select(corr("subscriber_count", "total_views")).show()
youtube_df.select(corr("total_videos", "total_views")).show()
youtube_df.select(corr("videos_per_week", "total_views")).show()
youtube_df.select(corr("mean_views_last_30_videos", "total_views")).show()
youtube_df.select(corr("median_views_last_30_videos", "total_views")).show()
youtube_df.select(corr("std_views_last_30_videos", "total_views")).show()



# COMMAND ----------

# MAGIC %md
# MAGIC ##### String correlations based on original (non-categorized) <i>total_views</i> column.

# COMMAND ----------

# MAGIC %sql
# MAGIC select country, total_views from youtube_data where total_views > 10000000000

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression and Classification Machine Learning Models Using a Pipeline

# COMMAND ----------

# Create a copy of the data for the machine learning model so as not to modify the original.

my_pipeline_data = youtube_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Create the Vector Assembler to contain the features including the One Hot Encoded <i>country</i> column.

# COMMAND ----------

from pyspark.ml.feature import (VectorAssembler, VectorIndexer, OneHotEncoder, StringIndexer)

# The String value field is country.  In order for these values to be used in a machine
# learning model, these String values need to be first converted to a float and then encoded into a vector using One Hot Encoder.

country_indexer = StringIndexer(inputCol='country', outputCol='countryIndex', handleInvalid = "keep")
country_encoder = OneHotEncoder(inputCol='countryIndex', outputCol='countryVector')

# Set the features for the data set.  These features will be the OHE encodings and the numerical fields that had 
# stronger correlations with the target, total_views or tvcat_indexed (total views category).

my_features = ['countryVector', 'subscriber_count', 'total_videos', 'videos_per_week', 'mean_views_last_30_videos', 'median_views_last_30_videos', 'std_views_last_30_videos' ] 

# Assemble the feature data together into one column.

assembler = VectorAssembler(inputCols=my_features, outputCol='features')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Regression and Classification Source Code
# MAGIC
# MAGIC Regression Models:
# MAGIC * Linear Regression
# MAGIC * Random Forest Regression
# MAGIC * Decision Tree Regression
# MAGIC * Gradient Boosted Trees (GBT) Regression
# MAGIC
# MAGIC Classification Models:
# MAGIC * Decision Tree Classifier
# MAGIC * Logistic Regression Classifier

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.regression import (LinearRegression, RandomForestRegressor, DecisionTreeRegressor, GBTRegressor)
from pyspark.ml.classification import (DecisionTreeClassifier, LogisticRegression)
from pyspark.ml.tuning import (CrossValidator, ParamGridBuilder)
from pyspark.ml.evaluation import (RegressionEvaluator, BinaryClassificationEvaluator, MulticlassClassificationEvaluator)
from pyspark.mllib.evaluation import (RegressionMetrics, MulticlassMetrics)


# Define global variables.

train_youtube_data = None
test_youtube_data = None
fit_model = None


# get_predictions_for_regression_model

def get_predictions_for_regression_model(modelType):

  global train_youtube_data
  global test_youtube_data
  global fit_model

  # Set the Target column.

  target = "total_views"

  # Create a Regression model using the Features and Target columns based on the modelType parameter.  

  rm_model = None
  paramGrid = None

  # Determine the Regression model to use.

  if (modelType == 'LinearRegression'):

    print ("Linear Regression Model")

    rm_model = (LinearRegression()
      .setLabelCol(target)
      .setFeaturesCol("features")
      .setElasticNetParam(0.5))
    
    paramGrid = ParamGridBuilder().addGrid(rm_model.maxIter, [500]) \
                                  .addGrid(rm_model.regParam, [0]) \
                                  .addGrid(rm_model.elasticNetParam, [1]) \
                                  .build()

  elif (modelType == 'RandomForest'):

    # Use Random Forest Regression.

    print ("Random Forest Regression Model")

    rm_model = RandomForestRegressor(labelCol=target, featuresCol="features", numTrees=10, maxDepth=3)

    paramGrid = (ParamGridBuilder()
      .addGrid(rm_model.numTrees, [10, 20])     # 100 1500
      .addGrid(rm_model.maxDepth, [5, 6])
      .build())
  
  elif (modelType == 'GBTRegression'):

    # Use GBT Regression.

    print ("GBT Regression Model")

    rm_model = GBTRegressor(labelCol=target, featuresCol="features")

    paramGrid = (ParamGridBuilder()
      .addGrid(rm_model.maxBins, [2])   
      .addGrid(rm_model.maxIter, [5])   
      .addGrid(rm_model.maxDepth, [5, 6])
      .build())

  else:

    # Use Decision Tree.

    print ("Decision Tree Regression Model")

    rm_model = DecisionTreeRegressor(labelCol=target, featuresCol="features")

    paramGrid = ParamGridBuilder() \
    .addGrid(rm_model.maxBins, [5]) \
    .addGrid(rm_model.maxDepth, [2, 5]) \
    .build()

  # Create the pipeline with the converted String indexes and the One Hot Encoding conversions.

  pipeline = Pipeline(stages=[country_indexer,
                              country_encoder,
                              assembler, rm_model])

  # Apply cross-validation to the pipeline.

  crossval = CrossValidator(estimator = pipeline,
                            estimatorParamMaps = paramGrid,
                            evaluator = RegressionEvaluator(labelCol = "total_views"),
                            numFolds = 3)

  # Partition the data into a 70/30 standard split.

  train_youtube_data, test_youtube_data = my_pipeline_data.randomSplit([0.7,.3])

  # Fit the training data into the model.  Use the cross validation instead of the pipeline.  

  fit_model = crossval.fit(train_youtube_data)

  # Make the predictions with the test data.

  predictions = fit_model.transform(test_youtube_data) 

  return predictions


# create_predictions_chart

def create_predictions_chart():

  # Create the predictions using the test data on the fitted training data model.
  # As predictions will be floating values, round them into double values to match the original price values.
  # From the exporatory data analysis, the data range of the total_views is 179,000,000,000.  
  # As this is a large range, it would be very rare given this data range for the prediction 
  # to exactly match the original value.
  # The median is 81,856 so this is a right-skewed distribution.
  # The median can't be used as part of predictions.  Therefore, let's look for an 80% - 120% range - that the predicted
  # target_views would be between 80% and 120% of the actual target_views - to indicate an accurate prediction.

  predictions_pipe = (fit_model
                      .transform(test_youtube_data)
                      .selectExpr("prediction as raw_prediction", 
                                  "double(round(prediction)) as prediction", 
                                  "double(total_views)", 
                                  """CASE ((double(round(prediction)) < (total_views * 1.20)) and (double(round(prediction)) > (total_views * .80))) 
                                  WHEN true then 1
                                  ELSE 0
                                  END as equal"""))
 
  return predictions_pipe


# display_regression_metrics

def display_regression_metrics(predictions_pipe):

  # Perform regression metrics based on the prediction and total_views.

  rm = RegressionMetrics(predictions_pipe.select("prediction", "total_views").rdd.map(lambda x:  (x[0], x[1])))
 
  print("MSE: ", rm.meanSquaredError)
  print("MAE: ", rm.meanAbsoluteError)
  print("RMSE Squared: ", rm.rootMeanSquaredError)
  print("R Squared: ", rm.r2)
  print("Explained Variance: ", rm.explainedVariance, "\n")


# get_predictions_for_classification_model

def get_predictions_for_classification_model(modelType):

  global train_youtube_data
  global test_youtube_data
  global fit_model

  # Set the Target column.

  target = "tvcat_indexed"

  # Create a Classification model using the Features and Target columns based on the modelType parameter.  

  class_model = None
  paramGrid = None

  # Determine the Classification model to use.

  if (modelType == 'DecisionTreeClassifier'):

    print ("Decision Tree Classification Model")

    class_model = DecisionTreeClassifier(labelCol=target, featuresCol="features")

    paramGrid = ParamGridBuilder() \
    .addGrid(class_model.maxDepth, [2, 5]) \
    .build()
  
  else:

    # Initialize the Logistic Regression classification model

    class_model = LogisticRegression(featuresCol="features", labelCol=target)

    paramGrid = ParamGridBuilder() \
    .addGrid(class_model.maxIter, [5]) \
    .build()


  # Create the pipeline with the converted String indexes and the One Hot Encoding conversions.

  pipeline = Pipeline(stages=[country_indexer,
                              country_encoder,
                              assembler, class_model])

  # Apply cross-validation to the pipeline.

  mc_evaluator = MulticlassClassificationEvaluator(labelCol="tvcat_indexed")

  crossval = CrossValidator(estimator = pipeline,
                            estimatorParamMaps = paramGrid,
                            evaluator = mc_evaluator,
                            numFolds = 3)

  # Partition the data into a 70/30 standard split.

  train_youtube_data, test_youtube_data = my_pipeline_data.randomSplit([0.7,.3])

  # Fit the training data into the model.  Use the cross validation instead of the pipeline.  

  fit_model = crossval.fit(train_youtube_data)

  # Make the predictions with the test data.

  predictions = fit_model.transform(test_youtube_data) 

  # Display the results.  Use MultiClass Metrics since the target classification has several classifications.

  area_under_curve = mc_evaluator.evaluate(predictions)
  print(f"Area under ROC curve: {area_under_curve}")

  lrmetrics = MulticlassMetrics(predictions['tvcat_indexed','prediction'].rdd)
  print('Confusion Matrix:\n', lrmetrics.confusionMatrix())
  print('F1 Score:', lrmetrics.fMeasure(1.0,1.0))
  print('False Positive Rate:', lrmetrics.falsePositiveRate(0.0))
  print('Precision:', lrmetrics.precision(1.0))
  print('Recall:', lrmetrics.recall(2.0))
  print('Accuracy:', lrmetrics.accuracy)

  return predictions


  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classification Machine Learning Models

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Tree Classifier

# COMMAND ----------

# Display the predictions.

display(get_predictions_for_classification_model('DecisionTreeClassifier'))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Logistic Regression Classifier

# COMMAND ----------

# Display the predictions.

display(get_predictions_for_classification_model('LogisticRegressionClassification'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression Machine Learning Models

# COMMAND ----------

# MAGIC %md
# MAGIC #### Linear Regression

# COMMAND ----------

# Display the predictions.

display(get_predictions_for_regression_model('LinearRegression'))

# COMMAND ----------

# Using the above data, create the predictions chart for the Linear Regression..

lg_predictions = create_predictions_chart()
display(lg_predictions)

# COMMAND ----------

# Determine the percentage of accurate predictions that were in +/- 20%.

display(lg_predictions.selectExpr("(sum(equal)/sum(1))*100"))


# COMMAND ----------

# Display the regression metrics for the predictions.

display_regression_metrics(lg_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Random Forest Regression

# COMMAND ----------

# Display the predictions.

display(get_predictions_for_regression_model('RandomForestRegression'))


# COMMAND ----------

# Create the predictions chart for the Random Forest Regression.

rfr_predictions = create_predictions_chart()
display(rfr_predictions)

# COMMAND ----------

# Determine the percentage of accurate predictions that were in +/- 20%.

display(rfr_predictions.selectExpr("(sum(equal)/sum(1))*100"))

# COMMAND ----------

# Display the regression metrics for the Random Forest Regression predictions.

display_regression_metrics(rfr_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Tree Regression

# COMMAND ----------

# Display the predictions.

display(get_predictions_for_regression_model('DecisionTreeRegression'))

# COMMAND ----------

# Create the predictions chart for the Random Forest Regression.

rfr_predictions = create_predictions_chart()
display(rfr_predictions)

# COMMAND ----------

# Determine the percentage of accurate predictions that were in +/- 20%.

display(rfr_predictions.selectExpr("(sum(equal)/sum(1))*100"))

# COMMAND ----------

# Display the regression metrics for the Random Forest Regression predictions.

display_regression_metrics(rfr_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC #### GBT Regression

# COMMAND ----------

# Display the predictions.

display(get_predictions_for_regression_model('GBTRegression'))


# COMMAND ----------

# Create the predictions chart for the Random Forest Regression.

rfr_predictions = create_predictions_chart()
display(rfr_predictions)

# COMMAND ----------

# Determine the percentage of accurate predictions that were in +/- 20%.

display(rfr_predictions.selectExpr("(sum(equal)/sum(1))*100"))

# COMMAND ----------

# Display the regression metrics for the Random Forest Regression predictions.

display_regression_metrics(rfr_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Summary
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Linear Regression is best performing Regression Model with an R-squared value of:  0.72  (closest to 1) and a Mean Absolute Error of 24,524,687 compared to a range of 179,000,000,000.
# MAGIC
# MAGIC Decision Tree Classifier is best performing Classification Model with an F1 score of 0.83, a False Positive Rate of 0.11, a Precision score of 0.87, a Recall score of: 0.54 and an accuracy rate of: 0.79.
# MAGIC
# MAGIC Between the two, Classification appears to perform better and is more accurate than Regression.
