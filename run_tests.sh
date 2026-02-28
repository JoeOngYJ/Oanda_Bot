#!/bin/bash
# Test runner script for oanda-trading-system
# This script handles ROS plugin conflicts by disabling plugin autoloading

# Set environment variable to disable automatic plugin loading
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

# Run pytest with ROS plugins explicitly disabled
python -m pytest "$@" \
    -p no:launch_testing_ros \
    -p no:launch_testing

# Exit with pytest's exit code
exit $?
