import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.util.MLUtils

object HelloRf
{
  val sc = new SparkContext("local", "HelloRf")

  // Load and parse the data file.
  // libsvm style iris Data - http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/iris.scale
  val data = MLUtils.loadLibSVMFile(sc, "/data/iris.scale")

  // Split the data into training and test sets (30% held out for testing)
  val splits = data.randomSplit(Array(0.7, 0.3))
  val (trainingData, testData) = (splits(0), splits(1))

  def main(args: Array[String]): Unit =
  {
    trainClassifier()
    trainRegressor()
  }

  def trainClassifier() =
  {
    val startTime = System.currentTimeMillis

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 4 // Iris data: 3 labels, (label + 1) value seems to be needed.
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 4 // <= 30
    val maxBins = 32

    val model = RandomForest.trainClassifier(
      trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.zipWithIndex.map
    {
      case(current, index) =>
        val predictionResult = model.predict(current.features)
        (index, current.label, predictionResult, current.label == predictionResult) // Tuple
    }

    val execTime = System.currentTimeMillis - startTime

    val testDataCount = testData.count()
    val testErrCount = labelAndPreds.filter(r => !r._4).count // r._4 = 4th element of tuple (current.label == predictionResult)
    val testSuccessRate = 100 - (testErrCount.toDouble / testDataCount * 100)

    println("RfClassifier Results: " + testSuccessRate + "% numTrees: " + numTrees + " maxDepth: " + maxDepth + " execTime(msec): " + execTime)
    println("Test Data Count = " + testDataCount)
    println("Test Error Count = " + testErrCount)
    println("Test Success Rate (%) = " + testSuccessRate)
    println("Learned classification forest model:\n" + model.toDebugString)
    labelAndPreds.foreach(x => println(x))
  }

  def trainRegressor()
  {
    val startTime = System.currentTimeMillis

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 0 // not used for regression
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 3 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "variance"
    val maxDepth = 4
    val maxBins = 32

    val model = RandomForest.trainRegressor(
      trainingData, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelsAndPredictions = testData.zipWithIndex.map
    {
      case(current, index) =>
        val predictionResult = model.predict(current.features)
        (index, current.label, predictionResult) // Tuple
    }

    val execTime = System.currentTimeMillis - startTime

    println("RfRegressor Results: execTime(msec): " + execTime)
    println("Learned regression forest model:\n" + model.toDebugString)
    labelsAndPredictions.foreach(x => println(x))
  }
}