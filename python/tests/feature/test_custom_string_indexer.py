import unittest
import os
import shutil
from feature import CustomStringIndexer, CustomStringIndexerModel
from tests import spark


class TestCustomStringIndexer(unittest.TestCase):

    def test_fitting_model(self):
        """
        'Bucketizer.fit()' return a model with the right fitted 'bins'
        """
        seq =  [(0, "aa"), (1, "ab"), (2, "cd"), (3, "ad"), (4, "aa")]
        dfm = spark.createDataFrame(seq).toDF("ID", "CATEGORY")

        indexer = CustomStringIndexer() \
            .setInputCol('CATEGORY') \
            .setOutputCol('CATEGORY_INDEX')

        model = indexer.fit(dfm)

        self.assertDictEqual(
            dict(model.mapping_cat),
            {'aa': 0, 'ab': 1, 'ad': 2, 'cd': 3}
        )

    def test_applying_fitted_model(self):
        """
        Bucketizer.transform should apply the correct transformation
        """
        seq =  [(0, "aa", 0.0), (1, "ab", 1.0), (2, "cd", 3.0),
                (3, "ad", 2.0), (4, "aa", 0.0)]
        dfm = spark.createDataFrame(seq).toDF("ID", "CATEGORY", "EXPECTED")

        indexer = CustomStringIndexer() \
            .setInputCol('CATEGORY') \
            .setOutputCol('CATEGORY_INDEX')

        dfm_trans = indexer.fit(dfm).transform(dfm)

        for row in dfm_trans.select('EXPECTED', 'CATEGORY_INDEX').collect():
            self.assertEqual(row.EXPECTED, row.CATEGORY_INDEX)

    def test_saving_model(self):
        """
        'Bucketizer.save(path)' should ensure we can load the model back into memory
        """
        temp_path = '.temp/'
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)

        seq =  [(0, "aa"), (1, "ab"), (2, "cd"), (3, "ad"), (4, "aa")]
        dfm = spark.createDataFrame(sq).toDF("ID", "CATEGORY")

        indexer = CustomStringIndexer() \
            .setInputCol('CATEGORY') \
            .setOutputCol('CATEGORY_INDEX')

        model = indexer.fit(dfm)
        df_transformed = model.transform(dfm)

        model.save(temp_path)

        loaded_model = CustomStringIndexerModel.load(temp_path)
        df_transformed_after_load = loaded_model.transform(dfm)

