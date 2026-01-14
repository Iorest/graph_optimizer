import unittest
import logging
from graph_optimizer.utils.logger import get_logger, set_log_level, INFO, DEBUG


class TestLogging(unittest.TestCase):
    def test_logger_setup(self):
        logger = get_logger("TestLogger")
        self.assertEqual(logger.name, "TestLogger")
        self.assertTrue(len(logger.handlers) >= 1)

    def test_set_log_level(self):
        set_log_level(DEBUG)
        from graph_optimizer.utils.logger import logger as global_logger

        self.assertEqual(global_logger.level, DEBUG)
        set_log_level(INFO)
        self.assertEqual(global_logger.level, INFO)


if __name__ == "__main__":
    unittest.main()
