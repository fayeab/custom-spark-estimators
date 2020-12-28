from pyspark import keyword_only
from pyspark.ml.param.shared import (Params, Param, TypeConverters,
                                     HasInputCol, HasOutputCol)
from pyspark.ml.util import JavaMLReadable, JavaMLWritable
from pyspark.ml.wrapper import JavaTransformer, JavaEstimator, JavaModel

class HasDropCol(Params):
    """
    Mixin for param column
    """

    column = Param(Params._dummy(), "column",
                      "Column to drop",
                       typeConverter=TypeConverters.toString)

    def __init__(self):
        super(HasDropCol, self).__init__()
        self._setDefault(column=None)

    def getDropCol(self):
        """
        Column to drop.
        """
        return self.getOrDefault(self.column)


class ColumnFilter(JavaTransformer, HasDropCol, JavaMLReadable, JavaMLWritable):
    """
    Filter column.
    """
    def __init__(self, column=None):
        """
        __init__(self, column=None)
        """
        super(ColumnFilter, self).__init__()
        self._java_obj = self._new_java_obj("feature.ColumnFilter", self.uid)
        self._setDefault(column=column)

    def setDropCol(self, value):
        """
        Sets the value of :py:attr:`column`.
        """
        return self._set(column=value)

class CustomStringIndexer(JavaEstimator, HasInputCol, HasOutputCol, JavaMLReadable, JavaMLWritable):

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(CustomStringIndexer, self).__init__()
        self._java_obj = self._new_java_obj("feature.CustomStringIndexer", self.uid)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    def _create_model(self, java_model):
        return CustomStringIndexerModel(java_model)


class CustomStringIndexerModel(JavaModel, JavaMLReadable, JavaMLWritable):
    """
    Model fitted by :py:class:`CustomStringIndexer`.
    """

    def setInputCol(self, value):
        """
        Sets the value of :py:attr:`inputCol`.
        """
        return self._set(inputCol=value)

    def setOutputCol(self, value):
        """
        Sets the value of :py:attr:`outputCol`.
        """
        return self._set(outputCol=value)

    @property
    def mapping_cat(self):
        """
        Containing mapping category -> index
        """
        return self._call_java("javaMappingCat")

