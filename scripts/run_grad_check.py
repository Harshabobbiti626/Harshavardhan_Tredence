"""
Single-command runner for Part 1: Automated finite-difference gradient checking
and masked-weight optimizer isolation test suite.
"""

import os
import sys
import pytest

# Ensure repository root is on sys.path regardless of execution directory
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

if __name__ == "__main__":
    print("=" * 80)
    print("AQUA PLATFORM - PART 1 GRADIENT CHECK & CORRECTNESS TEST SUITE")
    print("=" * 80)
    tests_dir = os.path.join(REPO_ROOT, "tests")
    args = ["-v", tests_dir] + sys.argv[1:]
    retcode = pytest.main(args)
    sys.exit(retcode)

