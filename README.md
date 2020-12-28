## <font color='blue'> Spark : Creation d'estimators et transformers </font>

Version: Spark 3.0.0

Avec scala:
```scala
import feature.CustomStringIndexer
val df = Seq(
    (0, "aa"), 
    (1, "ab"),
    (2, "cd"),
    (3, "ad"), 
    (4, "aa")
).toDF("ID", "CATEGORY")

val indexer = new CustomStringIndexer()
                  .setInputCol("CATEGORY")
                  .setOutputCol("CATEGORY_INDEX")

indexer.fit(df).transform(df).show()

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
from feature import CustomStringIndexer, CustomStringIndexerModel

seq =  [(0, "aa"), (1, "ab"), (2, "cd"), (3, "ad"), (4, "aa")]
dfm = spark.createDataFrame(seq).toDF("ID", "CATEGORY")
indexer = CustomStringIndexer()\
              .setInputCol('CATEGORY')\
              .setOutputCol('CATEGORY_FITTED')
model  = indexer.fit(dfm)
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
