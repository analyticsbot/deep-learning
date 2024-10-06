#!/bin/bash

# Start the SSH service in the background
service ssh start

# Keep the container running by using a long-running command
# This could be 'tail -f /dev/null' or any long-running process.
tail -f /dev/null
