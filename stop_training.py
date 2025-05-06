#!/usr/bin/env python
import os
import subprocess
import signal
import argparse

def find_training_processes():
    """Find all Python processes that are running train_Kfold_CV.py"""
    try:
        # Using ps command to find python processes
        cmd = "ps -ef | grep 'python.*train_Kfold_CV.py' | grep -v grep"
        result = subprocess.check_output(cmd, shell=True, text=True)
        
        processes = []
        for line in result.strip().split('\n'):
            if line:
                parts = line.split()
                pid = int(parts[1])  # Second field is PID
                cmd = ' '.join(parts[7:])  # Command is usually from the 8th field onwards
                processes.append((pid, cmd))
        
        return processes
    except subprocess.CalledProcessError:
        # No processes found
        return []

def kill_process(pid):
    """Kill a process by sending SIGTERM signal"""
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent termination signal to process {pid}")
        return True
    except ProcessLookupError:
        print(f"Process {pid} not found")
        return False
    except PermissionError:
        print(f"Permission denied for killing process {pid}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Stop training processes")
    parser.add_argument('--list-only', action='store_true', help='Only list processes without killing them')
    args = parser.parse_args()
    
    processes = find_training_processes()
    
    if not processes:
        print("No training processes found.")
        return
    
    print(f"Found {len(processes)} training processes:")
    for pid, cmd in processes:
        print(f"PID: {pid}, Command: {cmd}")
    
    if not args.list_only:
        confirm = input("Do you want to stop these processes? (y/n): ")
        if confirm.lower() == 'y':
            success_count = 0
            for pid, _ in processes:
                if kill_process(pid):
                    success_count += 1
            
            print(f"Successfully terminated {success_count} out of {len(processes)} processes")
        else:
            print("Operation cancelled")

if __name__ == "__main__":
    main() 