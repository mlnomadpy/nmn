#!/usr/bin/env python3
"""
Verification script for YAT implementations.

This script verifies that the YAT algorithm components are correctly implemented
by checking the mathematical formulations and code structure without requiring
actual TensorFlow/Keras dependencies.
"""

import os
import re
import sys


def verify_yat_algorithm_implementation(file_path):
    """Verify that a file contains the correct YAT algorithm implementation."""
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    checks = {
        'dot_product_computation': r'(dot_product|dot_prod_map|y).*=.*(ops\.matmul|tf\.matmul|conv)',
        'squared_computation': r'(inputs_squared|inputs.*\*\*.*2|tf\.square\(inputs\))',
        'distance_computation': r'(distance|distances).*=.*\+.*-.*2.*\*',
        'yat_computation': r'(dot_product|dot_prod_map|y|outputs).*(\*\*.*2|tf\.square).*\/.*\+.*epsilon',
        'alpha_scaling': r'(sqrt.*log|self\.alpha)',
        'kernel_squared': r'(kernel.*\*\*.*2|tf\.square.*kernel)',
    }
    
    passed_checks = []
    failed_checks = []
    
    for check_name, pattern in checks.items():
        if re.search(pattern, content, re.IGNORECASE):
            passed_checks.append(check_name)
        else:
            failed_checks.append(check_name)
    
    success = len(failed_checks) == 0
    return success, {
        'passed': passed_checks,
        'failed': failed_checks,
        'total_checks': len(checks)
    }


def verify_test_coverage(test_file_path):
    """Verify test coverage for YAT implementations."""
    if not os.path.exists(test_file_path):
        return False, f"Test file not found: {test_file_path}"
    
    with open(test_file_path, 'r') as f:
        content = f.read()
    
    test_functions = re.findall(r'def (test_\w+)', content)
    
    expected_tests = [
        'import',
        'basic',
        'no_bias',
        'no_alpha',
        'conv1d',
        'conv2d',
        'padding',
        'strides'
    ]
    
    covered_tests = []
    for expected in expected_tests:
        if any(expected in test_name for test_name in test_functions):
            covered_tests.append(expected)
    
    coverage = len(covered_tests) / len(expected_tests)
    
    return coverage >= 0.7, {
        'test_functions': test_functions,
        'covered_tests': covered_tests,
        'coverage': coverage
    }


def main():
    """Main verification function."""
    print("🔍 Verifying YAT Implementation Completeness")
    print("=" * 50)
    
    # Files to verify
    files_to_check = [
        ('Keras Dense', 'src/nmn/keras/nmn.py'),
        ('Keras Conv', 'src/nmn/keras/conv.py'),
        ('TensorFlow Dense', 'src/nmn/tf/nmn.py'),
        ('TensorFlow Conv', 'src/nmn/tf/conv.py'),
    ]
    
    test_files_to_check = [
        ('Keras Tests', 'tests/test_keras/test_keras_basic.py'),
        ('TensorFlow Tests', 'tests/test_tf/test_tf_basic.py'),
    ]
    
    all_passed = True
    
    # Verify implementations
    print("📋 Implementation Verification:")
    for name, file_path in files_to_check:
        success, result = verify_yat_algorithm_implementation(file_path)
        if success:
            print(f"✅ {name}: All YAT algorithm components present")
        else:
            print(f"❌ {name}: Missing components - {result['failed']}")
            all_passed = False
    
    print()
    
    # Verify test coverage
    print("🧪 Test Coverage Verification:")
    for name, test_file in test_files_to_check:
        success, result = verify_test_coverage(test_file)
        if success:
            print(f"✅ {name}: Good test coverage ({result['coverage']:.1%})")
            print(f"   Tests: {', '.join(result['test_functions'])}")
        else:
            print(f"❌ {name}: Insufficient test coverage ({result['coverage']:.1%})")
            all_passed = False
    
    print()
    
    # Check file structure
    print("📁 File Structure Verification:")
    required_files = [
        'src/nmn/keras/__init__.py',
        'src/nmn/tf/__init__.py',
        'src/nmn/keras/nmn.py',
        'src/nmn/keras/conv.py',
        'src/nmn/tf/nmn.py',
        'src/nmn/tf/conv.py',
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            all_passed = False
    
    print()
    
    # Summary
    if all_passed:
        print("🎉 All verifications passed! The YAT implementations appear complete.")
        return 0
    else:
        print("⚠️  Some verifications failed. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())