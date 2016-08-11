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
import scala.collection.mutable.ArrayBuffer

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
   * The following functions compute matrix vector multiplications for A^TB and B^TB.
   * We will use them to compute the spectral norms.
   * Note that this function is slightly different from that in OnePassPCA 
   * because of the different ways that A and B are stored.
   */
  def AtBv(rawData: RDD[(Int, (BDM[Double], BDM[Double]))], v: BDV[Double], n: Int, d: Int, sc: org.apache.spark.SparkContext): BDV[Double] = {
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
    AtBv
  }
  def BtAv(rawData: RDD[(Int, (BDM[Double], BDM[Double]))], v: BDV[Double], n: Int, d: Int, sc: org.apache.spark.SparkContext): BDV[Double] = {
    val vBC = sc.broadcast(v)
    val vtA = rawData.treeAggregate(BDV.zeros[Double](d))(
      (acc, value) => {
        val rowIndex = value._1
        val numRow = value._2._1.rows
        val vtA = vBC.value.slice(rowIndex*numRow, (rowIndex+1)*numRow).t*value._2._1
        brzAxpy(1.0, vtA.t, acc)
        acc},
      (acc1, acc2) => acc1+acc2
    )
    val vtABC = sc.broadcast(vtA)
    val BtAv = rawData.treeAggregate(BDV.zeros[Double](n))(
      (acc, value) => {
        val rowIndex = value._1
        val numRow = value._2._1.rows
        val Btv = value._2._2*vtABC.value
        acc(rowIndex*numRow to (rowIndex+1)*numRow-1) := Btv
        acc},
      (acc1, acc2) => acc1+acc2
    )
    BtAv
  }
  
   /*
    * The following function implements the (unnormalized) fast Walsh-Hadamard transform. 
    * The output is in 'sequency' order, note that any ordering works here
    * The length of the input vector is some power of 2
    * The code is adapted from Matlab's function 'fwht'
    */
   def fwht(v: Array[Double]): Array[Double] = {
    val N = v.size
    var x = Array.range(0,N).map{i=> if(i%2==0) v(i)+v(i+1) else v(i-1)-v(i)}
    var L = 1
    val numLog2 = 31 - Integer.numberOfLeadingZeros(N) 
    val y = Array.fill(N)(0.0)
    cfor(1)(i=>i<numLog2, i=>i+1){i=>
      var M = Math.pow(2,L).toInt
      var J = 0
      var K = 0
      while (K<N){
        cfor(J)(j=>j<J+M, j=>j+2){j=>
          y(K) = x(j)+x(j+M)
          y(K+1) = x(j)-x(j+M)
          y(K+2) = x(j+1)-x(j+1+M)
          y(K+3) = x(j+1)+x(j+1+M)
          K += 4
        }
        J += 2*M
      }
      x = y.clone()
      L += 1
    }
    y
    //(BDV(y):*=(1.0/Math.sqrt(N))).toArray //normalize the result, norm(y)=norm(v)
  }
    
  def main(args: Array[String]){
    //val args = Array("100000","100000","5","80","2000","10","0","2")
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
     * where G is a random Gaussian matrix, and D is a diagonal matrix with D_ii = 1/i.
     * Output: rawData is an RDD[rowIndex,(blockMatrixOfA, blockMatrixOfB)], 
     * and blockMatrixOfA has dimension rowsPerPart-by-d (which is slightly different from TwoPassLELA)
     */  
    println(s"Generate two random matrices A and B...")
    val rowsPerPart = n/p // number of rows per partition
    val alpha = 1
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
    println(s"Apply the Subsampled Randomized Hadamard Transform (SRHT) to the input...")
    val padZeros = if((d & -d)==d) d else 1<<(32-Integer.numberOfLeadingZeros(d)) // nearest power of 2
    val signBC = sc.broadcast(BDV.rand(d).map(x=> if(x>0.5) 1.0 else -1.0)) // random vector with +1/-1
    val indexBC = sc.broadcast(Random.shuffle(0 to padZeros-1).take(lowD).toArray) //random indices range from 0 to d-1
    val rawDataSketchRDD = rawData.mapValues{BDMs =>
      val numRow = BDMs._1.rows
      val rowSum = BDV(Array.tabulate(numRow)(i=> BDMs._1(i,::) dot BDMs._1(i,::)))
      val colnSum = BDV(Array.tabulate(numRow)(i=> BDMs._2(i,::) dot BDMs._2(i,::)))
      val BDMAsketch = new BDM(lowD, numRow, Array.tabulate(numRow){i=>
        val ht = fwht((BDMs._1(i,::).t:*signBC.value).toArray++Array.fill(padZeros-d)(0.0))
        val rescale = 1.0/Math.sqrt(lowD)
        indexBC.value.map(j=>ht(j)*rescale)
      }.flatten)
      val BDMBsketch = new BDM(lowD, numRow, Array.tabulate(numRow){i=>
        val ht = fwht((BDMs._2(i,::).t:*signBC.value).toArray++Array.fill(padZeros-d)(0.0))
        val rescale = 1.0/Math.sqrt(lowD)
        indexBC.value.map(j=>ht(j)*rescale)
      }.flatten)
      ((BDMAsketch.t, BDMBsketch.t), (rowSum, colnSum))
    }.cache()
    rawDataSketchRDD.count()

    println(s"Compute the norms...")
    val rowColnSum = rawDataSketchRDD.mapValues(x=>x._2).collectAsMap()
    val rowColnSumTilde = rawDataSketchRDD.mapValues{BDMs =>
      val numRow = BDMs._1._1.rows
      val rowSum = BDV(Array.tabulate(numRow)(i=> BDMs._1._1(i,::) dot BDMs._1._1(i,::)))
      val colnSum = BDV(Array.tabulate(numRow)(i=> BDMs._1._2(i,::) dot BDMs._1._2(i,::)))
      (rowSum, colnSum)
    }.collectAsMap()
    val rowSum = BDV(Array.tabulate(p)(i=>rowColnSum(i)._1.toArray).flatten)
    val colnSum = BDV(Array.tabulate(p)(i=>rowColnSum(i)._2.toArray).flatten)
    val rowSumTilde = BDV(Array.tabulate(p)(i=>rowColnSumTilde(i)._1.toArray).flatten)
    val colnSumTilde = BDV(Array.tabulate(p)(i=>rowColnSumTilde(i)._2.toArray).flatten)
 
    
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
    println(s"Collect sub-matrices of A and broadcast it; each node can then compute the sampled values...")
    val samplesBuf = new ArrayBuffer[Rating]
    val sampleLocationsBC = sc.broadcast(sampleLocations)
    val subMatNum = Math.min(Math.max(p/(d/lowD),1),p) // this parameter has to be large enough so that no "ByteStream.hugecapacity" error occurs
    val rowsPerSubMat = n/subMatNum
    val partsPerSubMat = rowsPerSubMat/rowsPerPart
   
    cfor(0)(i=>i<subMatNum, i=>i+1){i=>
      val partitions = Array.range(i*partsPerSubMat, (i+1)*partsPerSubMat).toSeq
      val res = sc.runJob(rawDataSketchRDD,
          (iter: Iterator[(Int,((BDM[Double], BDM[Double]),(BDV[Double],BDV[Double])))])=>iter.toArray,
          partitions)
      val subMat = res.map(x=>x(0)._2._1._1).foldLeft(BDM.zeros[Double](0,lowD))((x,y)=>BDM.vertcat(x,y))
      val subMatBC = sc.broadcast(subMat)
      val newSamples = rawDataSketchRDD.map{value=>
        val buf = new ArrayBuffer[Rating]
        cfor(0)(j=>j<rowsPerSubMat,j=>j+1){j=>
            val colnIndex = sampleLocationsBC.value(j+i*rowsPerSubMat)
            cfor(0)(k=>k<colnIndex.size,k=>k+1){k=>
              if(colnIndex(k)>=value._1*rowsPerPart && colnIndex(k)<(value._1+1)*rowsPerPart){
                buf += Rating(j+i*rowsPerSubMat, colnIndex(k), subMatBC.value(j,::) dot value._2._1._2(colnIndex(k)-value._1*rowsPerPart,::))
              }
            }
          }
        buf
      }.collect()
      cfor(0)(m=>m<newSamples.length, m=>m+1){m=>
        samplesBuf ++= newSamples(m)
      }
    }
    val samples = samplesBuf.toArray   
    val samplesRDD = sc.parallelize(samples).cache()
    val samNo = samplesRDD.count
    
    /*
     * Step 5: Run alternating minimization.
     */
    println(s"Run ALS on the samplesRDD...")
    val model22 = ALS.train(samplesRDD, r, Iterations, lambda)
    
    /*
     * Now we use AtBv and BtAv to compute the spectral norm error ||UV^T-A^TB||/||A^TB|| via power method
     */
    println(s"Compute the spectral norm ||A^TB||...")
    var normAB = 1.0  //the spectral norm of A^TB 
    val rand = new Random(System.currentTimeMillis)
    var v1 = BDV(Array.fill(n)(rand.nextGaussian))
    normAB = math.sqrt(v1 dot v1)
    v1 :*= 1/normAB
    cfor(0)(i=>i<3, i=>i+1){i=>
      v1 = AtBv(rawData, v1, n, d, sc)
      v1 = BtAv(rawData, v1, n, d, sc)
      normAB = math.sqrt(v1 dot v1)
      println(math.sqrt(normAB))
      v1 :*= 1/normAB
    }
    
    println(s"Compute the spectral norm ||UV^T-A^TB||...")
    val userBDM1 = featuresToBDM(model22.userFeatures,n,r)
    val prodBDM1 = featuresToBDM(model22.productFeatures,n,r)   
    var diffNorm = 1.0
    var v2 = BDV(Array.fill(n)(rand.nextGaussian))
    diffNorm = math.sqrt(v2 dot v2)
    v2 :*= 1/diffNorm
    cfor(0)(i=>i<5, i=>i+1){i=>
      val v3 = userBDM1*(v2.t*prodBDM1).t - AtBv(rawData, v2, n, d, sc)
      v2 = prodBDM1*(v3.t*userBDM1).t - BtAv(rawData, v3, n, d, sc)
      diffNorm = math.sqrt(v2 dot v2)
      println(math.sqrt(diffNorm))
      v2 :*= 1/diffNorm
    }
    val err = math.sqrt(diffNorm)/math.sqrt(normAB)
    println(s"The achieved spectral norm error is $err")

    sc.stop()   
  }
}

