import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.Vectors

object Cluster {
  def main(args: Array[String]) {    
    val conf = new SparkConf().setAppName("Cluster by Hiocde")
    val sc = new SparkContext(conf)
    val input_path= args(0)	//in hdfs
    val output_path= args(1)
    val temp_dic= args(2)
	  val numClusters = 23
	  val numIterations = 1000
	  val epsilon= 1e-4

    val input=sc.textFile(input_path)
    val matrix = input.map(s => Vectors.dense(s.split(' ').map(_.toDouble))).cache()
	
	  //val clusters = KMeans.train(matrix, numClusters, numIterations)
	  val model = new KMeans()
      .setK(numClusters)
      .setMaxIterations(numIterations)
      .setEpsilon(epsilon)
      .run(matrix)
    
    val res= model.predict(matrix)
    res.saveAsTextFile(output_path)
  }
}