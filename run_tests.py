
import unittest

import nearpy.tests as tests

suite = unittest.TestLoader().loadTestsFromModule(tests)
unittest.TextTestRunner(verbosity=2).run(suite)
