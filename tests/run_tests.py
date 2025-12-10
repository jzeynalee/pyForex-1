#!/usr/bin/env python3
"""
run_tests.py - Run all tests with various configurations
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_pytest(args):
    """Run pytest with given arguments."""
    cmd = [sys.executable, "-m", "pytest"] + args
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tests for pyForex")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--coverage", action="store_true", help="Generate coverage report")
    parser.add_argument("--html", action="store_true", help="Generate HTML report")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    pytest_args = []
    
    if args.unit:
        pytest_args.extend(["-m", "unit"])
    elif args.integration:
        pytest_args.extend(["-m", "integration"])
    
    if args.coverage:
        pytest_args.extend(["--cov=main", "--cov-report=term-missing"])
    
    if args.html:
        pytest_args.append("--html=test_report.html")
    
    if args.parallel:
        pytest_args.extend(["-n", "auto"])
    
    if args.verbose:
        pytest_args.append("-v")
    
    # Add test file if not already specified
    if not any(arg.startswith("test_") for arg in sys.argv):
        pytest_args.append("test_main.py")
    
    return run_pytest(pytest_args)


if __name__ == "__main__":
    sys.exit(main())