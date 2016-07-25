# MatrixProductPCA

## Overview
We provide two Spark implementations for the following problem: let `A` and `B` be two matrices of size d-by-n (assumed too large to fit in main memory), the goal is to find a rank-r approximation of their product A^TB in a pass-efficient way.

`LELA` is a two-pass algorithm, while `OnePassPCA` is a one-pass algorithm. The directory follows a typical layout: source file is located under `/src/main/scala`

The two-pass `LELA` is proposed by S. Bhojanapalli et al. in their paper [Tighter low-rank approximation via sampling the leveraged elements][LELA].

Current version: July 1, 2016 (More updates are expected soon.)

[LELA]: https://arxiv.org/abs/1410.3886

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
We are currently trying to speed up the current implementation of `OnePassPCA`, under the setting of large dense matrices and large sketch sizes.

* Replacing the standard JL (Gaussian matrix) by fast JL techniques (Hadamard sketch) could potentially reduce the time required for sketching. 
* Instead of collecting the sketched matrix RDD into the driver node and computing the samples locally, a better way may be to transpose the sketched matrix RDD (this involves global shuffling) and then compute the sampled values distributely.  
