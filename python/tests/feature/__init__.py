from pyspark import SparkConf
from pyspark.sql import SparkSession

conf = SparkConf().setAppName('Tests')
spark = SparkSession.builder.config(conf=conf).getOrCreate()