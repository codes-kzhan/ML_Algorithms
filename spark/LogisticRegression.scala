import java.util.Random

import scala.math.exp

import breeze.linalg.{Vector, DenseVector}

import org.apache.spark._
import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf


object Logistic {
  val iterations = 1
  val rand = new Random(42)
  
  case class DataPoint(x: Vector[Double], y: Double)
  
  def parsePoint(line: String): DataPoint = {
    var values = line.split(',')
    values(93) = values(93).last.toString()
    var values2 = values.map(_.toDouble)
    values2(93) = values2(93) - 1
    val x = Vector(values2.init)
    val y = {
      if (values2(93) == 0) -1
      else 1
    }
    DataPoint(x, y)
  }
  
  def score(p: DataPoint, w: DenseVector[Double]): Int = {
    val proba = 1 / (1 + exp(-w.dot(p.x)))
    val predict: Double = {
      if (proba > 0.5) 1.0
      else -1.0
    }
    if (predict == p.y) 1
    else 0
  }
  
  def main(args: Array[String]) {
    
    val conf = new SparkConf().setAppName("Logistic Regression")
    val sc = new SparkContext(conf)
    val csv_file = sc.textFile("/home/yejiming/desktop/spark/scala/dataset.csv")
    val data = csv_file.map(parsePoint)
    
    var w = DenseVector.fill(93){2 * rand.nextDouble - 1}
    println("Initial w: " + w)
    
    for (i <- 1 to iterations) {
      println("On iteration " + i)
      val gradient = data.map { p =>
        p.x * (1 / (1 + exp(-p.y * (w.dot(p.x)))) - 1) * p.y
      }.reduce((a, b) => a + b)
      w -= gradient
    }
    
    println("Final w: " + w)
    val correct_num: Double = data.map(p => score(p, w)).reduce(_ + _)
    val s: Double = correct_num / data.count()
    println("Prediction accuracy: " + s)
    
    sc.stop()
  }
}
