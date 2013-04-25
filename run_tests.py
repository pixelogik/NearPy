
import unittest

import nearpy.tests as tests

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestRandomBinaryProjections)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestEngine)
unittest.TextTestRunner(verbosity=2).run(suite)

suite = unittest.TestLoader().loadTestsFromTestCase(
    tests.TestStorage)
unittest.TextTestRunner(verbosity=2).run(suite)
