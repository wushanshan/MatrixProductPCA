# MatrixProductPCA

## Overview
We consider the following problem: let `A` and `B` be two matrices of size d-by-n (assumed too large to fit in main memory), the goal is to find a rank-r approximation of their product A^TB using a few passes over the data. A naive way is to compute A^TB first, and then perform truncated SVD. This algorithm suffers from O(n^2 d) time and O(n^2) memory and is hard to scale in the large-scale setting. An alternative option is to directly run power method without explicitly computing the product. This requires to access the data matrices O(r) times and potentially incur a huge disk IO overhead.

For this problem we present Spark implementations for two pass-efficient algorithms: `LELA` and `OnePassPCA`. Both algorithms requires approximately O(nrlogn) storage. `LELA` is a two-pass algorithm, proposed by S. Bhojanapalli et al. in their paper [Tighter low-rank approximation via sampling the leveraged elements][LELA]. `OnePassPCA`, as its name suggests, is a one-pass algorithm, and hence can be used when the matrices are coming from live data streams. The directory follows a typical layout: the source file is located under `/src/main/scala`

Note that: 1) For ease of computation, `A` and `B` are stored as a single RDD[index,(blockMatrixOfA, blockMatrixOfB)], and the RDDs are stored slightly differently: `LELA` stores row blocks while `OnePassPCA` stores column blocks. 2) [SRHT][srht] is implemented as the sketching step in `OnePassPCA`. It requires O(ndlogd) complexity, independent of the sketching size.

Current version: Aug 10, 2016.

[LELA]: https://arxiv.org/abs/1410.3886
[srht]: https://arxiv.org/abs/1011.1595

## Synthetic experiments

<img src="/images/runtime-3.png" width="450">


## How to run?
There are two ways to run in Spark: `spark-shell` or `spark-submit`.

### spark-shell
Launch the spark-shell (make sure you have [configured Spark][sparkConfig] properly), and then copy and paste our source code in the terminal.
[sparkConfig]: http://spark.apache.org/docs/latest/configuration.html

### spark-submit
First, build the JAR packge using `sbt package`. The generated JAR package is located under `/target/scala-2.10` Then copy the JAR pacakge to YOU_SPARK_HOME, and run application using spark-submit.

For example, the following scripts will run spark locally on 2 cores with memory 2g, with parameters #colns=5000, #rows=5000, rank=5, partitions=2, k=1000, #samples=4nrlogn. 

```$bin/spark-submit --class "LELA_ATB" --master local[2] --driver-memory 2g lela_2.10-1.0.jar 2000 2000 5 2 10 0 4```

```$bin/spark-submit --class "OnePassPCA" --master local[2] --driver-memory 2g onepasspca_2.10-1.0.jar 2000 2000 5 2 1000 10 0 4```


## Ongoing work
We are currently trying to speed up the Spark implementation of `OnePassPCA`, particularly for large dense matrices and large sketch sizes.

* Replacing the standard JL (Gaussian matrix) by fast JL techniques (Hadamard sketch) could potentially reduce the time required for sketching. 
* Instead of collecting the sketched matrix RDD into the driver node and computing the samples locally, a better way may be to transpose the sketched matrix RDD (this involves global shuffling) and then compute the sampled values distributely.  
