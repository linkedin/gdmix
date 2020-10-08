import unittest
from gdmixworkflow.single_node.local_ops import get_tfjob_cmd, get_sparkjob_cmd


class TestLocalOps(unittest.TestCase):
    """
    Test commands for single node workflow
    """

    def test_get_tfjob_cmd(self):
        params = {"--a": "b"}
        expected = ['python', '-m', 'gdmix.gdmix', "--a", "b"]
        actual = get_tfjob_cmd(params)
        self.assertEqual(actual, expected)

    def test_get_sparkjob_cmd(self):
        class_name = "Hello"
        params = {"-a": "b"}
        expected = ['spark-submit',
           '--class', "Hello",
           '--master', 'local[*]',
           '--num-executors','1',
           '--driver-memory', '1G',
           '--executor-memory', '1G',
           '--conf', 'spark.sql.avro.compression.codec=deflate',
           '--conf', 'spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs=false',
           'gdmix-data-all_2.11.jar',
           '-a', 'b']
        actual = get_sparkjob_cmd(class_name, params)
        self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
