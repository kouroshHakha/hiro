""" After running this unit test plot the test with the following command:
python plot.py unittest_out/logger -x step -y Averagevar2 Minvar3
"""

import time
import numpy as np
from logger import VectorLogger

from debug.cpdb import register_pdb
register_pdb()

class LoggerTester:

    def __init__(self):
        np.random.seed(10)

    def test_basics(self):
        logger = VectorLogger(output_dir='./unittest_out/logger/test_basics')
        s = time.time()

        for t in range(100):
            vec = np.random.randn(20).tolist()
            logger.log_tabular('step', t)
            logger.log_tabular('time', time.time() - s)
            logger.log_tabular('id', t // 10)
            logger.log_tabular('var1', float(np.random.randn()))
            logger.log_tabular('var2', vec)
            logger.log_tabular('var3', vec, with_min_and_max=True)
            logger.log_tabular('var4', vec, average_only=True)
            logger.log_tabular('var5', vec, with_min_and_max=True, average_only=True)
            logger.dump_tabular()


    def test_field_not_in_first_row(self):
        logger = VectorLogger(output_dir='./unittest_out/logger/test_field_not_in_first_row')
        s = time.time()

        for t in range(100):
            logger.log_tabular('step', t)
            logger.log_tabular('time', time.time() - s)

            if t >= 5 and t % 5 == 0:
                logger.log_tabular('var_5', float(np.random.randn()))
                print(logger.log_current_row)

            if t >= 10 and t % 10 == 0:
                logger.log_tabular('var_10', float(np.random.randn()))
                print(logger.log_current_row)

            logger.dump_tabular()

    def test_empty_log(self):
        logger = VectorLogger(output_dir='./unittest_out/logger/test_empty_log')
        for t in range(100):
            if t > 50:
                logger.log_tabular('t', t)
            logger.dump_tabular()

if __name__ == '__main__':

    unit = LoggerTester()
    unit.test_empty_log()
    unit.test_basics()
    unit.test_field_not_in_first_row()
