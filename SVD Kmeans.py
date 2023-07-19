import jieba 
import jieba.analyse 
import jieba.posseg as pseg 
from pyspark import SparkConf, SparkContext,SQLContext 
from pyspark.ml.feature import Word2Vec,CountVectorizer 

import pandas as pd 
from pyspark.ml.clustering import KMeans 
from pyspark.ml.feature import HashingTF, IDF, Tokenizer 
from pyspark.mllib.linalg.distributed import RowMatrix 
from pyspark.sql import Row 
from pyspark.ml.feature import VectorAssembler 
from pyspark.mllib.util import MLUtils 

conf = SparkConf().setAppName("cluster") 
sc = SparkContext(conf=conf) 
sqlContext=SQLContext(sc) 
#my_df 加载数据 
spark_df = sqlContext.createDataFrame(my_df) 

#计算tfidf 
cv = CountVectorizer(inputCol="words", outputCol="rawFeatures") 
cvmodel =cv.fit(spark_df); 
cvResult= cvmodel.transform(spark_df); 
idf = IDF(inputCol="rawFeatures", outputCol="features") 
idfModel = idf.fit(cvResult)  
cvResult = idfModel.transform(cvResult) 


ddf = MLUtils.convertVectorColumnsFromML(cvResult, 'features') 
ddf=ddf.select('features').rdd.map(lambda row : row[0]) 

mat = RowMatrix(ddf) 
#奇异值分解 
svd = mat.computeSVD(k=60, computeU=True) 
#转成dataframe格式 
svd_u = svd.U.rows.map(lambda row : row.tolist()) 
svd_df = sqlContext.createDataFrame(svd_u) 
#kmeans聚类 
kmeans = KMeans().setK(60).setSeed(1) 
vecAssembler = VectorAssembler(inputCols=svd_df.schema.names, outputCol='features') 
svd_df = vecAssembler.transform(svd_df) 
#聚类结果 
c_result = svd_df.select('features') 
model = kmeans.fit(c_result) 
results = model.transform(svd_df)