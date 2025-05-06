#!/bin/bash

# Function to find and display training processes
find_processes() {
  echo "Searching for training processes..."
  procs=$(ps -ef | grep 'python.*train_Kfold_CV.py' | grep -v grep)
  
  if [ -z "$procs" ]; then
    echo "No training processes found."
    exit 0
  fi
  
  echo "Found the following training processes:"
  echo "$procs"
}

# Function to kill the processes
kill_processes() {
  pids=$(ps -ef | grep 'python.*train_Kfold_CV.py' | grep -v grep | awk '{print $2}')
  
  if [ -z "$pids" ]; then
    echo "No training processes found to kill."
    exit 0
  fi
  
  for pid in $pids; do
    echo "Stopping process with PID: $pid"
    kill -15 $pid  # SIGTERM
  done
  
  echo "Checking if processes are still running..."
  sleep 2
  
  # Check if any of the processes are still running
  still_running=$(ps -ef | grep 'python.*train_Kfold_CV.py' | grep -v grep)
  if [ -n "$still_running" ]; then
    echo "Some processes are still running. Force killing them..."
    for pid in $pids; do
      kill -9 $pid 2>/dev/null  # SIGKILL, suppress errors for already terminated processes
    done
  else
    echo "All training processes have been terminated."
  fi
}

# Main script
if [ "$1" == "--list-only" ]; then
  find_processes
else
  find_processes
  echo ""
  read -p "Do you want to stop these processes? (y/n): " answer
  
  if [ "$answer" == "y" ] || [ "$answer" == "Y" ]; then
    kill_processes
  else
    echo "Operation cancelled."
  fi
fi 