package feature

import com.spark.util.{DefaultParamsReader, DefaultParamsWriter}

import org.apache.hadoop.fs.Path
import org.apache.spark.ml.param.{Param, Params, ParamMap}
import org.apache.spark.ml.Transformer
import org.apache.spark.ml.util.{DefaultParamsReadable, DefaultParamsWritable,
                                 Identifiable, MLWritable, MLWriter, MLReadable, MLReader}
import org.apache.spark.sql.{Dataset, DataFrame}
import org.apache.spark.sql.types.StructType


trait ColumnFilterParams extends Params {

  /**
   * Param for regularization parameter (&gt;= 0).
   * @group param
   */
  val column: Param[String] = new Param[String](this, "column", "Column to Drop")

  //setDefault(featuresCol, "features")

  /** @group getParam */
  def getDropCols: String = $(column)

  /** Validates and transforms the input schema. */
  protected def validateAndTransformSchema(schema: StructType): StructType = {
    require(isDefined(column), s"CustomStringIndexer requires input column parameter: $column")
    schema
  }

}

class ColumnFilter(override val uid: String) extends Transformer with ColumnFilterParams
  with DefaultParamsReadable[ColumnFilter] with DefaultParamsWritable with MLWritable{

  def this() = this(Identifiable.randomUID("colfilter"))

  /** @group setParam */
  def setDropCols(value: String): this.type =  set(column, value)

  override def transformSchema(schema: StructType): StructType = {
    validateAndTransformSchema(schema)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    dataset.drop(this.getDropCols)
  }

  override def copy(extra: ParamMap): ColumnFilter = defaultCopy(extra)

  override def toString: String = {
    s"ColumnFilter: uid=$uid\nDrop: " + this.getDropCols
  }

  override def write: MLWriter = new ColumnFilter.ColumnFilterWriter(this)

}

object ColumnFilter extends MLReadable[ColumnFilter] {

  class ColumnFilterWriter(instance: ColumnFilter) extends MLWriter {

    private case class Data(column: String)

    override protected def saveImpl(path: String): Unit = {
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      val data = Data(instance.getDropCols)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)
    }
  }

  private class ColumnFilterReader extends MLReader[ColumnFilter] {

    private val className = classOf[ColumnFilter].getName

    override def load(path: String): ColumnFilter = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)
      val dataPath = new Path(path, "data").toString
      val data = sparkSession.read.parquet(dataPath)
        .select("column")
        .head()

      val column = data.getAs[String](0)

      val transformer = new ColumnFilter(metadata.uid).setDropCols(column)
      metadata.getAndSetParams(transformer)
      transformer
    }
  }

  override def read: MLReader[ColumnFilter] = new ColumnFilterReader

  override def load(path: String): ColumnFilter = super.load(path)


}




