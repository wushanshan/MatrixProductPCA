# MatrixProductPCA

## Overview
We consider the following problem: let `A` and `B` be two matrices of size d-by-n (assumed too large to fit in main memory), the goal is to find a rank-r approximation of their product A^TB using a few passes over the data. A naive way is to compute A^TB first, and then perform truncated SVD. This algorithm suffers from O(n^2 d) time and O(n^2) memory and is hard to scale in the large-scale setting. An alternative option is to directly run power method without explicitly computing the product. This requires to access the data matrices O(r) times and potentially incur a huge disk IO overhead.

For this problem we present Spark implementations for two pass-efficient algorithms: `LELA` and `OnePassPCA`. `LELA` is a two-pass algorithm, proposed by S. Bhojanapalli et al. in their paper [Tighter low-rank approximation via sampling the leveraged elements][LELA]. `OnePassPCA`, as its name suggests, is a one-pass algorithm, and hence can be used when the matrices are coming from live data streams. It implements the algorithm `SMP-PCA` (which stands for Streaming Matrix Product PCA) proposed in our paper [Single Pass PCA of Matrix Prodcuts][1passLELA]. 

__Note__: Both of `OnePassPCA` and `SMP-PCA` refer to the same single-pass algorithm. `OnePassPCA` is the old name. 

The directory follows a typical layout: the source file is located under `/src/main/scala`

Here are two flow diagrams illustrating `LELA` (upper) and `OnePassPCA` (lower).

<img src="/images/flow-diagram.png" width="650"> 

__Note__: 1) For ease of computation, `A` and `B` are stored as a single RDD[index,(blockMatrixOfA, blockMatrixOfB)]. The RDDs are stored in a slightly different way: `LELA` stores row blocks while `OnePassPCA` stores column blocks. 2) [SRHT][srht] is implemented as the sketching step in `OnePassPCA`. It requires O(ndlogd) complexity, independent of the sketching size.

Current version: Aug 10, 2016.

[LELA]: https://arxiv.org/abs/1410.3886
[srht]: https://arxiv.org/abs/1011.1595
[1passLELA]: https://arxiv.org/abs/1610.06656

## Synthetic experiments
We now present simulation results on a 150GB synthetic example: let n=d=100000, r=5, the matrices `A` and `B` are generated as `DG`, where `G` has entries independently drawn from standard Gaussian distribution and `D` is a diagonal matrix with D_ii = 1/i. Other parameters are set as #RDD partitions=80, #samples = 2nrlogn, sketching size = 2000, and #ALS iterations = 10. 

We run Spark-1.6.2 on an Amazon EC2 cluster with two [m3.2xlarge][aws] instances. We use the [spark-ec2][ec2] script to lauch clusters. The results are shown in the following table. Here we compare the relative spectral norm error, which is calculated as ||A^TB- UV^T||/||A^TB||.  

|    Methods |  Error    |  Runtime |
|----------- |-----------|----------|
|  Exact SVD |  0.0271   |  23 hrs  |
|    LELA    |  0.0274   |  56mins  |
| OnePassPCA |  0.0280   |  34mins  |

__Note__: 1) For `Exact SVD`, we adapt the source code of private object [EigenValueDecomposition][SVD] for our setting: compute B^TAA^TBv distributedly and send it to ARPACK's dsaupd to compute the top eigenvalues and eigenvectors. We set tol=1e-5 as the convergence tolerance. 2)  Sometimes we will encounter the error `"Remote RPC client disassociated"`. The exact cause is still unknown.

[SVD]:https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/mllib/linalg/EigenValueDecomposition.scala

The following figure illustrates the runtime breakdown on clusters of 2, 5, and 10 nodes. Each node is an [m3.2xlarge][aws] instance. We see that the speedup achieved by OnePassPCA is more prominent for small clusters (possibly due to the increasing spark overheads at larger clusters, see [this paper][ov] for more explanation on spark overheads). 

<img src="/images/runtime-breakdown.png" width="450"> 

[aws]:https://aws.amazon.com/ec2/pricing/on-demand/
[ec2]:http://spark.apache.org/docs/1.6.2/ec2-scripts.html
[ov]: https://arxiv.org/abs/1607.01335

## How to run our code?
There are two ways to run in Spark: `spark-shell` or `spark-submit`. 

### spark-shell
Launch the spark-shell (make sure your Spark is [configured][sparkConfig] properly), and then copy and paste our source code in the terminal.
[sparkConfig]: http://spark.apache.org/docs/latest/configuration.html

### spark-submit
Build the JAR packge using `sbt package`. The generated JAR package is located under `/target/scala-2.10` Then copy the JAR pacakge to YOU_SPARK_HOME, and run application using spark-submit.

For example, the following scripts will run spark locally on 2 cores with memory 2g, with parameters #colns=5000, #rows=5000, rank=5, partitions=2, sketching size=1000, ALS iterations=10, ALS lambda=0, #samples=2nrlogn. 

```$bin/spark-submit --class "LELA_ATB" --master local[2] --driver-memory 2g lela_2.10-1.0.jar 5000 5000 5 2 10 0 2```

```$bin/spark-submit --class "OnePassPCA" --master local[2] --driver-memory 2g onepasspca_2.10-1.0.jar 5000 5000 5 2 1000 10 0 2```

## Matlab implementation
We also provide Matlab code for the two algorithms. Note that the provided Matlab code uses standard JL (Gaussian matrix) for the sketching step, while our Spark code implementes Hadamard sketch ([SRHT][srht]) which runs faster.
