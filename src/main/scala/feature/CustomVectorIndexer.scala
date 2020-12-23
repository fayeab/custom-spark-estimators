package feature

import java.lang.{Double => JDble, String => JStr}
import java.util.{Map => JMap, NoSuchElementException}
import scala.collection.JavaConverters._

import com.spark.util.{DefaultParamsReader, DefaultParamsWriter}
import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable, Identifiable, MLReadable, MLReader, MLWritable, MLWriter}
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.{DoubleType, StringType, StructField, StructType}
import org.apache.spark.sql.functions.{col, udf}
import org.apache.spark.broadcast.Broadcast


trait CustomVectorIndexerParams extends Params {

  /**
   * Param for regularization parameter (&gt;= 0).
   * @group param
   */
  val inputCol: Param[String] = new Param[String](this, "inputCol", "Input column name")
  val outputCol: Param[String] = new Param[String](this, "outputCol", "Output column name")

  //setDefault(featuresCol, "features")

  /** @group getParam */
  def getInputCol: String = $(inputCol)
  def getOutputCol: String = $(outputCol)

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(isDefined(inputCol), s"CustomVectorIndexer requires input column parameter: $inputCol")
    require(isDefined(outputCol), s"CustomVectorIndexer requires output column parameter: $outputCol")

    val field = schema.fields(schema.fieldIndex($(inputCol)))

    if (field.dataType!= StringType) {
      throw new Exception(
        s"Input type ${field.dataType} did not match input type DoubleType")
    }

    // Add the return field
    schema.add(StructField($(outputCol), DoubleType, false))
  }

}

class CustomVectorIndexerModel (
                                 override val uid: String,
                                 mappingCat: Array[(String, Double)]
                               )
  extends Model[CustomVectorIndexerModel] with CustomVectorIndexerParams
    with DefaultParamsReadable[CustomVectorIndexerModel]  with DefaultParamsWritable
    with MLWritable {

  /** @group setParam */
  def setInputCol(value: String): this.type = set(inputCol, value)

  /** @group setParam */
  def setOutputCol(value: String): this.type = set(outputCol, value)

  /** @group getParam */
  def getMappingCat: Array[(String, Double)] = mappingCat

  val mapCat = this.getMappingCat.toMap.asInstanceOf[Map[String, Double]]
  private var mapCatBroadcast: Option[Broadcast[Map[String, Double]]] = None

  def javaMappingCat: JMap[JStr, JDble] = {
    mapAsJavaMap(mapCat).asInstanceOf[JMap[JStr, JDble]]
  }

  override def transform(dataset: Dataset[_]): DataFrame = {

    transformSchema(dataset.schema, logging = true)

    if (mapCatBroadcast.isEmpty) {
         mapCatBroadcast = Some(dataset.sparkSession.sparkContext.broadcast(mapCat))
    }

    def transformCat(
                      mapped: Map[String, Double],
                      value: String
                    ): Double = mapped.get(value).getOrElse(-1).asInstanceOf[Int].toDouble

    def getTransformFunc(
                          mapped: Map[String, Double]
                        ): String => Double = {value: String => transformCat(mapped, value)}

    val maps = mapCatBroadcast.get.value
    val func = getTransformFunc(maps)
    val transformer = udf(func)

    dataset.withColumn(this.getOutputCol, transformer(col(this.getInputCol)))

  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }
  override def copy(extra: ParamMap): CustomVectorIndexerModel = {
    val copied = new CustomVectorIndexerModel(uid,
      mappingCat)
    copyValues(copied, extra).setParent(parent)
  }

  override def write: MLWriter = new CustomVectorIndexerModel.CustomVectorIndexerModelWriter(this)

  override def toString: String = {
    s"CustomVectorIndexerModel: uid=$uid"
  }
}


object CustomVectorIndexerModel extends MLReadable[CustomVectorIndexerModel] {

  class CustomVectorIndexerModelWriter(instance: CustomVectorIndexerModel) extends MLWriter {

    private case class Data(mapping: Seq[(String, Integer)])

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val seqMapping = instance.getMappingCat.toSeq.asInstanceOf[Seq[(String, Integer)]]
      val data = Data(seqMapping)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(data.mapping).repartition(1).write.parquet(dataPath)
    }
  }

  private class CustomVectorIndexerModelReader extends MLReader[CustomVectorIndexerModel] {

    private val className = classOf[CustomVectorIndexerModel].getName

    override def load(path: String): CustomVectorIndexerModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
      val maps = data.collect().toList.map{e => {(e(0), e(1))}}
        .toArray
        .asInstanceOf[Array[(String, Double)]]
      val model = new CustomVectorIndexerModel(metadata.uid, maps)
      metadata.getAndSetParams(model)
      model
    }
  }

  override def read: MLReader[CustomVectorIndexerModel] = new CustomVectorIndexerModelReader

  override def load(path: String): CustomVectorIndexerModel = super.load(path)
}

class CustomVectorIndexer (override val uid: String)
  extends Estimator[CustomVectorIndexerModel] with CustomVectorIndexerParams
    with DefaultParamsReadable[CustomVectorIndexer]
    with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("cstri"))

  /** @group setParam */

  def setInputCol(value: String): this.type = set(inputCol, value)

  def setOutputCol(value: String): this.type = set(outputCol, value)


  override def fit(dataset: Dataset[_]): CustomVectorIndexerModel = {
    val mappingCat = dataset.select($(inputCol))
      .distinct().collect()
      .map(_(0)).toArray
      .sortBy(_.asInstanceOf[String]).zipWithIndex
      .asInstanceOf[Array[(String, Double)]]
    copyValues(new CustomVectorIndexerModel(uid,
      mappingCat).setParent(this))

  }

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def copy(extra: ParamMap): CustomVectorIndexer = defaultCopy(extra)

}
