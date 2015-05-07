import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import scala.collection.mutable.ListBuffer

object logistic {
    
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
        val training = splits(0).cache()
        val test = splits(1)

        // Run training algorithm to build the model
        val model = new LogisticRegressionWithLBFGS().setNumClasses(9).run(training)

        // Compute raw scores on the test set.
        val predictionAndLabels = test.map { 
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
