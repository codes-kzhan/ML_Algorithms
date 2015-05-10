import scala.collection.mutable.HashMap
import scala.collection.mutable.HashSet

import breeze.linalg.{Vector, DenseVector, squaredDistance}

import org.apache.spark.SparkContext._
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf


object Kmeans {
  val K = 9
  val convergeDist = 10
  
  def parsePoint(line: String): Vector[Double] = {
    var values = line.split(',')
    values(93) = values(93).last.toString()
    var values2 = values.map(_.toDouble)
    values2(93) = values2(93) - 1
    Vector(values2.init)
  }
  
  def distance(p1: Vector[Double], p2: Vector[Double]): Double = {
    var dist = 0.0
    
    for (i <- 0 until p1.size) {
      dist += (p2(i) - p1(i)) * (p2(i) - p1(i))
    }
    
    dist
  }
    
  def closestPoint(p: Vector[Double], centers: HashMap[Int, Vector[Double]]): Int = {
    var index = 0
    var bestIndex = 0
    var closest = Double.PositiveInfinity
    
    for (i <- 1 to centers.size) {
      val vCurr = centers.get(i).get
      val tempDist = distance(p, vCurr)
      if (tempDist < closest) {
        closest = tempDist
        bestIndex = i
      }
    }  
    
    bestIndex 
  }
  
  def main(args: Array[String]) {
    
    val conf = new SparkConf().setAppName("Kmeans Clustering")
    val sc = new SparkContext(conf)
    val csv_file = sc.textFile("/home/yejiming/desktop/spark/scala/dataset.csv")
    val data = csv_file.map(parsePoint)
    var points = new HashSet[Vector[Double]]
    var kPoints = new HashMap[Int, Vector[Double]]
    var tempDist = 11.0
    
    while (points.size < K) {
      points.add(data.takeSample(false, 1)(0))
    }
    
    val iter = points.iterator
    for (i <- 1 to points.size) {
      kPoints.put(i, iter.next())
    }
    
    println("Initial centers: " + kPoints)
    
    while(tempDist > convergeDist) {
      var closest = data.map(p => (closestPoint(p, kPoints), (p, 1)))
      
      var pointStats = closest.reduceByKey ( 
        (a, b) => (a._1 + b._1, a._2 + b._2)
      )
      
      var newPoints = pointStats.map { mapping =>
        (mapping._1, mapping._2._1 * (1.0 / mapping._2._2))
      }
      
      val newPoints2 = newPoints.toArray
      
      tempDist = 0.0
      for (mapping <- newPoints2) {
        tempDist += distance(kPoints.get(mapping._1).get, mapping._2)
        println("current improvement: " + tempDist)
      }
      
      for (newP <- newPoints2) {
        kPoints.put(newP._1, newP._2)
      }
      
    }
    
    println("Final centers: " + kPoints)
  }
}
