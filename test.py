#!/usr/bin/env python3
"""
Dummy test script for CI/CD pipeline.
This script does nothing but print a message and always passes.
"""
import unittest

class DummyTest(unittest.TestCase):
    def test_dummy(self):
        """A dummy test that always passes and prints a message."""
        print("Dummy test only")
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
