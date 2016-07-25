/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.spark.mllib.linalg.distributed._
import org.apache.spark.mllib.linalg._
import scala.util.Random
import org.apache.spark.mllib.recommendation._
import org.apache.spark.rdd.RDD
import org.apache.spark.storage._
import org.apache.spark.{SparkConf, SparkContext}
import breeze.linalg.{DenseVector => BDV, SparseVector => BSV, Vector => BV, 
  axpy=>brzAxpy, sum, trace, accumulate=>cumsum, DenseMatrix => BDM, Axis}
import breeze.numerics.{round}
import spire.syntax.cfor._

object OnePassPCA { 
  /*
   * Transform IndexRowMatrix to BDM of size n-by-r
   */
  def indexRowToBDM(A: org.apache.spark.mllib.linalg.distributed.IndexedRowMatrix, n: Int, r: Int): BDM[Double]={
      new BDM(n,r, A.toBlockMatrix().toLocalMatrix().toArray)
  }
  
  /*
   * Transform featureRDD to BDM of size n-by-r
   */
  def featuresToBDM(A: RDD[(Int, Array[Double])], n: Int, r: Int): BDM[Double]={
      indexRowToBDM(new IndexedRowMatrix(A.map(x=>new IndexedRow(x._1, Vectors.dense(x._2)))), n, r)
  }
  
  /*
   * The following function samples from a distribution with a CDF proportional to 
   * cumsum(0), cumsum(1)+shift, cumsum(2)+2*shift,...cumsum(n) + n*shift
   * Sampling are performed by first generate a random number in [0,1], 
   * and then locate its column index by binary searching.
   * The output is an iterator over sampled indices. 
   */
  def sampleColns(cumsum: BDV[Double], shift: Double, n: Int, SampleNo: Int):Iterator[Int] = {      
    val res = BSV.zeros[Boolean](n)
    cfor(0)(i=>i<SampleNo, i=>i+1){i=>
      val x = Random.nextFloat()
      var a = 0
      var b = n
      val mx = cumsum(n) + n*shift
      while (b-a>1) {
        val mid = (a + b)/2
        val temp = (cumsum(mid) + mid*shift)/mx
        if (x<=temp) b = mid else a = mid
      }
      if (res(a)==false) {
        res(a) = true
      }
    }
    res.activeKeysIterator
  }
  
  /*
   *  The following function computes matrix vector multiplication for A^TB.
   *  It outputs B^TAA^TBv, and we will use it to get the spectral norm of A^TB.
   *  Note that this function is slightly different from that in TwoPassLELA
   *  because of the different ways that A and B are stored.
   */
  def matVecMulti(rawData: RDD[(Int, (BDM[Double], BDM[Double]))], v: BDV[Double], n: Int, d: Int, sc: org.apache.spark.SparkContext): BDV[Double] = {
    val vBC = sc.broadcast(v)
    val vtB = rawData.treeAggregate(BDV.zeros[Double](d))(
      (acc, value) => {
        val rowIndex = value._1
        val numRow = value._2._1.rows
        val vtB = vBC.value.slice(rowIndex*numRow, (rowIndex+1)*numRow).t*value._2._2
        brzAxpy(1.0, vtB.t, acc)
        acc},
      (acc1, acc2) => acc1+acc2
    )
    val vtBBC = sc.broadcast(vtB)
    val AtBv = rawData.treeAggregate(BDV.zeros[Double](n))(
      (acc, value) => {
        val rowIndex = value._1
        val numRow = value._2._1.rows
        val Atv = value._2._1*vtBBC.value
        acc(rowIndex*numRow to (rowIndex+1)*numRow-1) := Atv
        acc},
      (acc1, acc2) => acc1+acc2
    )
    val AtBvBC = sc.broadcast(AtBv)
    val AAtBv = rawData.treeAggregate(BDV.zeros[Double](d))(
      (acc, value) => {
        val rowIndex = value._1
        val numRow = value._2._1.rows
        val vtA = AtBvBC.value.slice(rowIndex*numRow, (rowIndex+1)*numRow).t*value._2._1
        brzAxpy(1.0, vtA.t, acc)
        acc},
      (acc1, acc2) => acc1+acc2
    )
    val AAtBvBC = sc.broadcast(AAtBv)
    val BtAAtBv = rawData.treeAggregate(BDV.zeros[Double](n))(
      (acc, value) => {
        val rowIndex = value._1
        val numRow = value._2._1.rows
        val Btv = value._2._2*AAtBvBC.value
        acc(rowIndex*numRow to (rowIndex+1)*numRow-1) := Btv
        acc},
      (acc1, acc2) => acc1+acc2
    )
    BtAAtBv
  }
  
  def main(args: Array[String]){
    //val args = Array("50000","50000","5","25","1000","10","0","4")
    val n = args(0).toInt // number of columns of A & B
    val d = args(1).toInt // number of rows of A & B
    val r = args(2).toInt // desired rank
    val p = args(3).toInt // number of partitions for an RDD
    val lowD = args(4).toInt // sketching dimension
    val Iterations = args(5).toInt // number of iterations for running ALS
    val lambda = args(6).toDouble //regularization parameter for ALS
    val ss = args(7).toDouble //#samples = ss*nrlog(n)
    
    println(s"Step 1: running OnePassPCA with #colns=$n, #rows=$d, rank=$r, partitions=$p, k=$lowD, samples=$ss")
    
    val sparkConf = new SparkConf().setAppName("OnePassPCA")
    val sc = new SparkContext(sparkConf)
    
    /*
     * Generates two random matrices A and B of the form D*G, 
     * where G is a random Gaussian matrix, and D is a diagonal matrix with D_ii = 1/\sqrt(i).
     * Output: rawData is an RDD[rowIndex,(blockMatrixOfA, blockMatrixOfB)], 
     * and blockMatrixOfA has dimension rowsPerPart-by-d (which is slightly different from TwoPassLELA)
     */  
    println(s"Generate two random matrices A and B...")
    val rowsPerPart = n/p // number of rows per partition
    val alpha = 0.5
    val rawData = sc.parallelize(Array.range(0,p),p).map{x=> 
      val rng = new Random(x).self
      val BDMA = new BDM(rowsPerPart,d, DenseMatrix.randn(rowsPerPart, d,rng).toArray)
      val BDMB = new BDM(rowsPerPart,d, DenseMatrix.randn(rowsPerPart, d,rng).toArray)
      cfor(0)(i=>i<BDMA.cols, i=>i+1){i=>
           BDMA(::,i) :*= 1.0/math.pow(i+1,alpha)
           BDMB(::,i) :*= 1.0/math.pow(i+1,alpha)
        }
      (x,(BDMA, BDMB))
    }
    rawData.persist(StorageLevel.DISK_ONLY)
    rawData.count()
   
    
    /*
     * Step 2: Build sketches and compute column norms (actual_norm and norm_tilde)
     * Output: rawDataSketch: Map[rowIndex, (blockMatA_tilde, blockMatB_tilde)];
     * rowSum, rowSumTilde: BDV, column norm of matrix A and A_tilde; 
     * colnSum, colnSumTilde: BDV, column norm of matrix B and B_tilde; 
     */
    println(s"Step 2: Build sketches and compute column norms...")
    println(s"Generate a sketching matrix and broadcast it to all nodes...")
    val now = System.currentTimeMillis
    val rng = new Random(now).self
    val randomMat = new BDM(d,lowD,DenseMatrix.randn(d, lowD, rng).toArray.map(x=>x/math.sqrt(lowD)))
    val randomMatBC = sc.broadcast(randomMat) // limited to 2GB size by Spark
    
    println(s"Compute the sketches...")
    val rawDataSketchRDD =  rawData.mapValues{BDMs =>
      val numRow = BDMs._1.rows
      val rowSum = BDV(Array.tabulate(numRow)(i=> BDMs._1(i,::) dot BDMs._1(i,::)))
      val colnSum = BDV(Array.tabulate(numRow)(i=> BDMs._2(i,::) dot BDMs._2(i,::)))
     ((BDMs._1*randomMatBC.value, BDMs._2*randomMatBC.value), (rowSum, colnSum))
    }.cache()
    rawDataSketchRDD.count()
    val rawDataSketch = rawDataSketchRDD.mapValues(x=>x._1).collectAsMap()
    
    println(s"Compute the norms...")
    val rowColnSum = rawDataSketchRDD.mapValues(x=>x._2).collectAsMap()
    val rowColnSumTilde = rawDataSketchRDD.mapValues{BDMs =>
      val numRow = BDMs._1._1.rows
      val rowSum = BDV(Array.tabulate(numRow)(i=> BDMs._1._1(i,::) dot BDMs._1._1(i,::)))
      val colnSum = BDV(Array.tabulate(numRow)(i=> BDMs._1._2(i,::) dot BDMs._1._2(i,::)))
      (rowSum, colnSum)
    }.collectAsMap()
    var rowSum = BDV.rand(0)
    var colnSum = BDV.rand(0)
    var rowSumTilde = BDV.rand(0)
    var colnSumTilde = BDV.rand(0)
    cfor(0)(i=>i<p,i=>i+1){i=>
      rowSum = BDV.vertcat(rowSum, rowColnSum(i)._1)
      colnSum = BDV.vertcat(colnSum, rowColnSum(i)._2)
      rowSumTilde = BDV.vertcat(rowSum, rowColnSumTilde(i)._1)
      colnSumTilde = BDV.vertcat(colnSum, rowColnSumTilde(i)._2)
    }
    
    /*
     * Step 3: Perform biased sampling.
     * Output: sampleLocations: Map(rowIndex->SampledColumns)
     */
    println(s"Step3: perform biased sampling...")
    println(s"Compute #samples per row...")
    val sampleNo = ss*math.ceil(n*r*math.log(n)).toInt // total no. of samples   
    val rowSumPDF =  BDV(Array.fill(n)(1.0/(2*n)))   
    val fNorm_A = sum(rowSum)
    val fNorm_B = sum(colnSum)
    brzAxpy(1.0/(2*fNorm_A), rowSum, rowSumPDF) 
    val numberOfSamplesPerRow = Array.tabulate(n)(i=>(i,round(rowSumPDF(i)*sampleNo).toInt )) // Array((Int, Int)) 
    val normalizedColnSum = BDV(colnSum.toArray)
    normalizedColnSum:*=(0.5/(n*fNorm_B)) 
    val colnCumsum = BDV.vertcat(BDV(0.0),cumsum(normalizedColnSum))
    val normalizedRowSum = BDV(rowSum.toArray)
    normalizedRowSum:*=(0.5/(n*fNorm_A))
   
    println(s"Sample numberOfSamplesPerRow(i) for row-i...")
    val colnCumsumBC = sc.broadcast(colnCumsum) // BDV[Double]
    val colnSumBC = sc.broadcast(normalizedColnSum) //BDV[Double]
    val rowSumBC = sc.broadcast(normalizedRowSum) //BDV[Double]
    val sampleLocations = sc.parallelize(numberOfSamplesPerRow).map{ case (rowIndex, numSam)=>
      val sampleIndex = if (numSam*math.log(n)/math.log(2)>n){
        val colnSumPDF = colnSumBC.value
        brzAxpy(1.0, rowSumBC.value, colnSumPDF)
        colnSumPDF:*=(1/sum(colnSumPDF))
        val res = BSV.zeros[Boolean](n)
        cfor(0)(i=>i<n, i=>i+1){i=>
           res(i) = math.random<numSam*colnSumPDF(i)
        }
        res.active.keys.iterator
      } else {
        sampleColns(colnCumsumBC.value,rowSumBC.value(rowIndex), n, numSam)
      }   
      (rowIndex, sampleIndex.toArray)
    }.collectAsMap() 
    
    /*
     * Step 4: Construct the sampled matrix.
     * Output: RDD[Rating(rowIndex, colnIndex, value)]
     */
    println(s"Calculate the sampled values using scala parallel collection...")
    val samples = sampleLocations.toList.par.map{v=>
      val value = Array.fill(v._2.size)(0.0)
      val vecA = rawDataSketch(v._1/rowsPerPart)._1(v._1-v._1/rowsPerPart*rowsPerPart,::)
      cfor(0)(i=>i<v._2.size, i=>i+1){i=> 
        val vecB = rawDataSketch(v._2(i)/rowsPerPart)._2(v._2(i)-v._2(i)/rowsPerPart*rowsPerPart,::)
        value(i) = (vecA dot vecB)*math.sqrt(rowSum(v._1)*colnSum(v._2(i)))/math.sqrt(rowSumTilde(v._1)*colnSumTilde(v._2(i)))
      }
      (v._1, v._2.zip(value))
    }.toList
    val samplesRDD = sc.parallelize(samples).flatMap{case (rowIndex, samples)=>
      samples.map(x=>Rating(rowIndex,x._1, x._2))
    }.cache()
    val samNo = samplesRDD.count
    
    /*
     * Step 5: Run alternating minimization.
     */
    println(s"Run ALS on the samplesRDD...")
    val model22 = ALS.train(samplesRDD, r, Iterations, lambda)
    
    /*
     * Now we use power method to compute the spectral norm error ||UV^T-A^TB||/||A^TB||.
     */
    println(s"Compute the spectral norm ||A^TB||...")
    var normAB = 1.0  //the spectral norm of A^TB 
    val normal = breeze.stats.distributions.Gaussian(0,1)
    var v1 = BDV(normal.sample(d).toArray)
    normAB = math.sqrt(v1 dot v1)
    v1 :*= 1/normAB
    cfor(0)(i=>i<5, i=>i+1){i=>
      v1 = matVecMulti(rawData, v1, n, d, sc)
      normAB = math.sqrt(v1 dot v1)
      println(math.sqrt(normAB))
      v1 :*= 1/normAB
    }
    
    println(s"Compute the spectral norm ||UV^T-A^TB||...")
    val userBDM1 = featuresToBDM(model22.userFeatures,n,r)
    val prodBDM1 = featuresToBDM(model22.productFeatures,n,r)   
    var diffNorm = 1.0
    var v2 = BDV(normal.sample(d).toArray)
    diffNorm = math.sqrt(v2 dot v2)
    v2 :*= 1/diffNorm
    cfor(0)(i=>i<5, i=>i+1){i=>
      val v3 = userBDM1*(v2.t*prodBDM1).t
      v2 = prodBDM1*(v3.t*userBDM1).t - matVecMulti(rawData, v2, n, d, sc)
      diffNorm = math.sqrt(v2 dot v2)
      println(math.sqrt(diffNorm))
      v2 :*= 1/diffNorm
    }
    val err = math.sqrt(diffNorm)/math.sqrt(normAB)
    println(s"The achieved spectral norm error is $err")
    
    sc.stop()   
  }
}

