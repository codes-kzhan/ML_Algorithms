import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.ListBuffer

object forest {
    
    def parsePoint(line: String): LabeledPoint = {
        var values = line.split(',')
        values(93) = values(93).last.toString()
        var values2 = values.map(_.toDouble)
        values2(93) = values2(93) - 1
        LabeledPoint(values2.last, Vectors.dense(values2.init))
    }

    def main(args: Array[String]) {
        val conf = new SparkConf().setAppName("logistic regression")
        val sc = new SparkContext(conf)

        // Load training data in csv format.
        val csv_file = sc.textFile("/home/yejiming/desktop/spark/scala/dataset.csv")
        val data = csv_file.map(parsePoint)

        // Split data into training (80%) and test (20%).
        val splits = data.randomSplit(Array(0.8, 0.2), seed = 123)
        val (trainingData, testData) = (splits(0), splits(1))

        // Setting up the hyper-parameters
        val numClasses = 9
        val categoricalFeaturesInfo = Map[Int, Int]()
        val numTrees = 50 // Use more in practice.
        val featureSubsetStrategy = "auto" // Let the algorithm choose.
        val impurity = "gini"
        val maxDepth = 10
        val maxBins = 100

        // Run training algorithm to build the model
        val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
             numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

        // Compute raw scores on the test set.
        val predictionAndLabels = testData.map { 
            case LabeledPoint(label, features) =>
            val prediction = model.predict(features)
            (prediction, label)
        }

        // Get evaluation metrics.
        val metrics = new MulticlassMetrics(predictionAndLabels)
        val precision = metrics.precision
        println("Precision = " + precision)
    }
}
