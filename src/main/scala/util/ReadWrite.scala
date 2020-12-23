package com.spark.util


import org.apache.hadoop.fs.Path
import org.json4s._
import org.json4s.{DefaultFormats, JObject}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

import org.apache.spark.SparkContext
import org.apache.spark.ml.param.{ParamPair, Params}
import org.apache.spark.ml.util.{MLWriter, MLReadable, MLReader}

object Utils {

  def getSparkClassLoader: ClassLoader = getClass.getClassLoader

  def getContextOrSparkClassLoader: ClassLoader =
    Option(Thread.currentThread().getContextClassLoader).getOrElse(getSparkClassLoader)
    Option(Thread.currentThread().getContextClassLoader).getOrElse(getSparkClassLoader)

  def classForName(className: String): Class[_] = {
    Class.forName(className, true, getContextOrSparkClassLoader)
  }

}

/**
 * Default `MLWriter` implementation for transformers and estimators that contain basic
 * (json4s-serializable) params and no data. This will not handle more complex params or types with
 * data (e.g., models with coefficients).
 *
 * @param instance object to save
 */
class DefaultParamsWriter(instance: Params) extends MLWriter {

  override protected def saveImpl(path: String): Unit = {
    DefaultParamsWriter.saveMetadata(instance, path, sc)
  }
}

object DefaultParamsWriter {

  /**
   * Saves metadata + Params to: path + "/metadata"
   *  - class
   *  - timestamp
   *  - sparkVersion
   *  - uid
   *  - defaultParamMap
   *  - paramMap
   *  - (optionally, extra metadata)
   *
   * @param extraMetadata  Extra metadata to be saved at same level as uid, paramMap, etc.
   * @param paramMap  If given, this is saved in the "paramMap" field.
   *                  Otherwise, all [[org.apache.spark.ml.param.Param]]s are encoded using
   *                  [[org.apache.spark.ml.param.Param.jsonEncode()]].
   */
  def saveMetadata(
                    instance: Params,
                    path: String,
                    sc: SparkContext,
                    extraMetadata: Option[JObject] = None,
                    paramMap: Option[JValue] = None): Unit = {
    val metadataPath = new Path(path, "metadata").toString
    val metadataJson = getMetadataToSave(instance, sc, extraMetadata, paramMap)
    sc.parallelize(Seq(metadataJson), 1).saveAsTextFile(metadataPath)
  }

  /**
   * Helper for [[saveMetadata()]] which extracts the JSON to save.
   * This is useful for ensemble models which need to save metadata for many sub-models.
   *
   * @see [[saveMetadata()]] for details on what this includes.
   */
  def getMetadataToSave(
                         instance: Params,
                         sc: SparkContext,
                         extraMetadata: Option[JObject] = None,
                         paramMap: Option[JValue] = None): String = {
    val uid = instance.uid
    val cls = instance.getClass.getName
    val params = instance.extractParamMap().toSeq.asInstanceOf[Seq[ParamPair[Any]]]
    val jsonParams = paramMap.getOrElse(render(params.map { case ParamPair(p, v) =>
      p.name -> parse(p.jsonEncode(v))
    }.toList))

    val basicMetadata = ("class" -> cls) ~
      ("timestamp" -> System.currentTimeMillis()) ~
      ("sparkVersion" -> sc.version) ~
      ("uid" -> uid) ~
      ("paramMap" -> jsonParams)
    val metadata = extraMetadata match {
      case Some(jObject) =>
        basicMetadata ~ jObject
      case None =>
        basicMetadata
    }
    val metadataJson: String = compact(render(metadata))
    metadataJson
  }
}

/**
 * Default `MLReader` implementation for transformers and estimators that contain basic
 * (json4s-serializable) params and no data. This will not handle more complex params or types with
 * data (e.g., models with coefficients).
 *
 * @tparam T ML instance type
 * TODO: Consider adding check for correct class name.
 */

class DefaultParamsReader[T] extends MLReader[T] {

  override def load(path: String): T = {
    val metadata = DefaultParamsReader.loadMetadata(path, sc)
    val cls = Utils.classForName(metadata.className)
    val instance =
      cls.getConstructor(classOf[String]).newInstance(metadata.uid).asInstanceOf[Params]
    metadata.getAndSetParams(instance)
    instance.asInstanceOf[T]
  }
}

object DefaultParamsReader {

  /**
   * All info from metadata file.
   *
   * @param params  paramMap, as a `JValue`
   * @param defaultParams defaultParamMap, as a `JValue`. For metadata file prior to Spark 2.4,
   *                      this is `JNothing`.
   * @param metadata  All metadata, including the other fields
   * @param metadataJson  Full metadata file String (for debugging)
   */
  case class Metadata(
                       className: String,
                       uid: String,
                       timestamp: Long,
                       sparkVersion: String,
                       params: JValue,
                       defaultParams: JValue,
                       metadata: JValue,
                       metadataJson: String) {


    private def getValueFromParams(params: JValue): Seq[(String, JValue)] = {
      params match {
        case JObject(pairs) => pairs
        case _ =>
          throw new IllegalArgumentException(
            s"Cannot recognize JSON metadata: $metadataJson.")
      }
    }

    /**
     * Get the JSON value of the [[org.apache.spark.ml.param.Param]] of the given name.
     * This can be useful for getting a Param value before an instance of `Params`
     * is available. This will look up `params` first, if not existing then looking up
     * `defaultParams`.
     */
    def getParamValue(paramName: String): JValue = {
      implicit val format = DefaultFormats

      // Looking up for `params` first.
      var pairs = getValueFromParams(params)
      var foundPairs = pairs.filter { case (pName, jsonValue) =>
        pName == paramName
      }
      if (foundPairs.length == 0) {
        // Looking up for `defaultParams` then.
        pairs = getValueFromParams(defaultParams)
        foundPairs = pairs.filter { case (pName, jsonValue) =>
          pName == paramName
        }
      }
      assert(foundPairs.length == 1, s"Expected one instance of Param '$paramName' but found" +
        s" ${foundPairs.length} in JSON Params: " + pairs.map(_.toString).mkString(", "))

      foundPairs.map(_._2).head
    }

    /**
     * Extract Params from metadata, and set them in the instance.
     * This works if all Params (except params included by `skipParams` list) implement
     * [[org.apache.spark.ml.param.Param.jsonDecode()]].
     *
     * @param skipParams The params included in `skipParams` won't be set. This is useful if some
     *                   params don't implement [[org.apache.spark.ml.param.Param.jsonDecode()]]
     *                   and need special handling.
     */
    def getAndSetParams(
                         instance: Params,
                         skipParams: Option[List[String]] = None): Unit = {
      setParams(instance, skipParams, isDefault = false)

    }

    private def setParams(
                           instance: Params,
                           skipParams: Option[List[String]],
                           isDefault: Boolean): Unit = {
      implicit val format = DefaultFormats
      val paramsToSet = if (isDefault) defaultParams else params
      paramsToSet match {
        case JObject(pairs) =>
          pairs.foreach { case (paramName, jsonValue) =>
            if (skipParams == None || !skipParams.get.contains(paramName)) {
              val param = instance.getParam(paramName)
              val value = param.jsonDecode(compact(render(jsonValue)))
              instance.set(param, value)
            }
          }
        case _ =>
          throw new IllegalArgumentException(
            s"Cannot recognize JSON metadata: ${metadataJson}.")
      }
    }
  }

  /**
   * Load metadata saved using [[DefaultParamsWriter.saveMetadata()]]
   *
   * @param expectedClassName  If non empty, this is checked against the loaded metadata.
   * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
   */
  def loadMetadata(path: String, sc: SparkContext, expectedClassName: String = ""): Metadata = {
    val metadataPath = new Path(path, "metadata").toString
    val metadataStr = sc.textFile(metadataPath, 1).first()
    parseMetadata(metadataStr, expectedClassName)
  }

  /**
   * Parse metadata JSON string produced by [[DefaultParamsWriter.getMetadataToSave()]].
   * This is a helper function for [[loadMetadata()]].
   *
   * @param metadataStr  JSON string of metadata
   * @param expectedClassName  If non empty, this is checked against the loaded metadata.
   * @throws IllegalArgumentException if expectedClassName is specified and does not match metadata
   */
  def parseMetadata(metadataStr: String, expectedClassName: String = ""): Metadata = {
    val metadata = parse(metadataStr)

    implicit val format = DefaultFormats
    val className = (metadata \ "class").extract[String]
    val uid = (metadata \ "uid").extract[String]
    val timestamp = (metadata \ "timestamp").extract[Long]
    val sparkVersion = (metadata \ "sparkVersion").extract[String]
    val defaultParams = metadata \ "defaultParamMap"
    val params = metadata \ "paramMap"
    if (expectedClassName.nonEmpty) {
      require(className == expectedClassName, s"Error loading metadata: Expected class name" +
        s" $expectedClassName but found class name $className")
    }

    Metadata(className, uid, timestamp, sparkVersion, params, defaultParams, metadata, metadataStr)
  }

  /**
   * Load a `Params` instance from the given path, and return it.
   * This assumes the instance implements [[MLReadable]].
   */
  def loadParamsInstance[T](path: String, sc: SparkContext): T =
    loadParamsInstanceReader(path, sc).load(path)

  /**
   * Load a `Params` instance reader from the given path, and return it.
   * This assumes the instance implements [[MLReadable]].
   */
  def loadParamsInstanceReader[T](path: String, sc: SparkContext): MLReader[T] = {
    val metadata = DefaultParamsReader.loadMetadata(path, sc)
    val cls = Utils.classForName(metadata.className)
    cls.getMethod("read").invoke(null).asInstanceOf[MLReader[T]]
  }
}

