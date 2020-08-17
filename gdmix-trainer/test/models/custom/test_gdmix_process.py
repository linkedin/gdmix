import tensorflow as tf
from gdmix.models.custom.scipy.gdmix_process import GDMixProcess


class TestGDMixProcess(tf.test.TestCase):
    """
    Unit tests for GDMixProcess, which is a wrapper over multiprocessing Process
    """

    def method_to_add_numbers(self, num1, num2):
        """
        Sample method to trigger successful run using GDMixProcess
        """
        return num1 + num2

    def method_to_throw_exception(self, num1, num2):
        """
        Sample method to trigger failure while using GDMixProcess
        """
        raise Exception("Method threw exception")

    def test_gdmixprocess_should_run_without_exception_for_successful_run(self):

        # Run a GDMixProcess for a method that works
        sample_process = GDMixProcess(target=self.method_to_add_numbers, args=(4, 5,))
        sample_process.start()
        sample_process.join()

        # Assert no exception is attached to process
        self.assertIsNone(sample_process.exception)

    def test_train_should_fail_if_producer_or_consumer_fails2(self):

        # Run a GDMixProcess for a method that fails
        sample_process = GDMixProcess(target=self.method_to_throw_exception, args=(4, 5,))
        sample_process.start()
        sample_process.join()

        # Assert exception is attached. Note that the process doesn't bubble up exception, but stores it as an
        # instance field
        self.assertIsNotNone(sample_process.exception)
