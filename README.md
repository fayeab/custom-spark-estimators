## <font color='blue'> Spark : Creation d'estimators et transformers </font>

Version: Spark 3.0.0

Avec scala:
```scala
import feature.CustomVectorIndexer
val df = Seq(
    (0, "aa"), 
    (1, "ab"),
    (2, "cd"),
    (3, "ad"), 
    (4, "aa")
).toDF("ID", "CATEGORY")

val vectorizer = new CustomVectorIndexer()
                  .setInputCol("CATEGORY")
                  .setOutputCol("CATEGORY_INDEX")

vectorizer.fit(df).transform(df).show()

```
```
+---+--------+--------------+
| ID|CATEGORY|CATEGORY_INDEX|
+---+--------+--------------+
|  0|      aa|           0.0|
|  1|      ab|           1.0|
|  2|      cd|           3.0|
|  3|      ad|           2.0|
|  4|      aa|           0.0|
+---+--------+--------------+
```

Avec python:

```python
from feature import CustomVectorIndexer, CustomVectorIndexerModel

seq =  [(0, "aa"), (1, "ab"), (2, "cd"), (3, "ad"), (4, "aa")]
dfm = spark.createDataFrame(seq).toDF("ID", "CATEGORY")
vectorizer = CustomVectorIndexer()\
              .setInputCol('CATEGORY')\
              .setOutputCol('CATEGORY_FITTED')
model  = vectorizer.fit(dfm)
model.transform(dfm).show(truncate=False)

```
```
+---+--------+---------------+
|ID |CATEGORY|CATEGORY_FITTED|
+---+--------+---------------+
|0  |aa      |0              |
|1  |ab      |1              |
|2  |cd      |3              |
|3  |ad      |2              |
|4  |aa      |0              |
+---+--------+---------------+
```


```python

```
