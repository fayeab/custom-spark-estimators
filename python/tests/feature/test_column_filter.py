import unittest
import os
import shutil
from feature import ColumnFilter
from tests import spark

class TestColumnFilter(unittest.TestCase):

    def test_transform(self):
        """
        ColumnFilter.transform should apply the correct transformation
        """
        seq =  [(0, "aa"), (1, "ab"), (2, "cd"),
                (3, "ad"), (4, "aa")]
        dfm = spark.createDataFrame(seq).toDF("ID", "CATEGORY")

        dropper = ColumnFilter().setDropCol('ID')

        dfm_trans = dropper.transform(dfm)

        self.assertListEqual(list(dfm_trans.columns), ["CATEGORY"])

    def test_saving_model(self):
        """
        'Bucketizer.save(path)' should ensure we can load the model back into memory
        """
        temp_path = '.temp/'
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        seq =  [(0, "aa"), (1, "ab"), (2, "cd"), (3, "ad"), (4, "aa")]
        dfm = spark.createDataFrame(seq).toDF("ID", "CATEGORY")

        dropper = ColumnFilter().setDropCol('ID')

        dfm_trans = dropper.transform(dfm)

        model.save(temp_path)

        loaded_model = ColumnFilter.load(temp_path)
        df_transformed_after_load = loaded_model.transform(dfm)
