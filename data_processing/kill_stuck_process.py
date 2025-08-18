# -------------------------------------------------------------------------
# File: kill_stuck_process.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-03-2025
# File Path: data_processing/kill_stuck_process.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   This module provides emergency process management functionality to terminate
#   stuck or hanging data processing operations when normal interruption methods
#   (like Ctrl+C) fail to work. It serves as a safety mechanism to forcefully
#   terminate unresponsive processes related to the APH-IF data processing pipeline,
#   including LangChain operations, OpenAI API calls, and Neo4j database operations.
#   The module helps maintain system stability by preventing resource exhaustion
#   from hung processes.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: find_stuck_processes()
# - Function: kill_process()
# - Function: main()
# - Constants: None
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os (environment operations), sys (system operations), signal (process signaling)
# - Third-Party: psutil (cross-platform process and system utilities)
# - Local Project Modules: None (standalone emergency utility script)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module is designed to be run as an emergency standalone script when
# data processing operations become unresponsive. It can be executed directly
# from the command line to identify and terminate stuck processes. The module
# is typically used as a last resort when normal process termination methods
# fail to work effectively.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

Emergency Process Termination for APH-IF Data Processing

Provides emergency functionality to identify and terminate stuck data processing
processes when normal interruption methods fail, ensuring system stability and
preventing resource exhaustion from hanging operations.

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
#!/usr/bin/env python3
import os  # Operating system interface for environment operations
import sys  # System-specific parameters and functions
import signal  # Signal handling for process control
from pathlib import Path  # Object-oriented filesystem path handling

# Third-party library imports
import psutil  # Cross-platform process and system utilities

# Local application/library specific imports
# None - this is a standalone emergency utility script


# =========================================================================
# Global Constants / Variables
# =========================================================================
# No global constants defined in this module


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# These are functions that are not methods of any specific class within this module.

# --------------------------
# --- Utility Functions ---
# --------------------------

# --------------------------------------------------------------------------------- find_stuck_processes()
def find_stuck_processes():
    """Identifies data processing processes that may be stuck or hanging.

    Scans all running processes to find those related to APH-IF data processing
    operations, including processes running data processing scripts, LangChain
    operations, and OpenAI API calls. Analyzes CPU usage patterns to identify
    potentially stuck processes.

    The function looks for processes with command lines containing keywords
    related to data processing operations and evaluates their CPU usage to
    determine if they might be hanging.

    Returns:
        list: A list of dictionaries containing information about potentially
            stuck processes. Each dictionary contains:
            - 'pid' (int): Process ID
            - 'name' (str): Process name
            - 'cmdline' (str): Command line (truncated to 100 characters)
            - 'cpu_percent' (float): CPU usage percentage
            - 'create_time' (float): Process creation timestamp

    Examples:
        >>> stuck_procs = find_stuck_processes()
        >>> for proc in stuck_procs:
        ...     print(f"PID {proc['pid']}: {proc['name']} - CPU: {proc['cpu_percent']:.1f}%")
        PID 1234: python - CPU: 0.0%
    """
    stuck_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time', 'cpu_percent']):
        try:
            cmdline = ' '.join(proc.info['cmdline']) if proc.info['cmdline'] else ''
            
            # Look for data processing related processes
            if any(keyword in cmdline.lower() for keyword in [
                'run_data_processing',
                'initial_graph_build',
                'launch_data_processing',
                'langchain',
                'openai'
            ]):
                # Check if process might be stuck (low CPU usage)
                cpu_percent = proc.cpu_percent(interval=1)
                
                stuck_processes.append({
                    'pid': proc.info['pid'],
                    'name': proc.info['name'],
                    'cmdline': cmdline[:100] + '...' if len(cmdline) > 100 else cmdline,
                    'cpu_percent': cpu_percent,
                    'create_time': proc.info['create_time']
                })
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return stuck_processes
# --------------------------------------------------------------------------------- end find_stuck_processes()

# --------------------------------------------------------------------------------- kill_process()
def kill_process(pid, force=False):
    """Terminates a process by its Process ID with optional force killing.

    Attempts to gracefully terminate a process using SIGTERM, and if that fails
    or if force is requested, uses SIGKILL for immediate termination. Provides
    detailed feedback about the termination process and handles various error
    conditions that may occur during process termination.

    Args:
        pid (int): The Process ID of the process to terminate.
        force (bool, optional): If True, immediately use SIGKILL instead of
            trying SIGTERM first. Defaults to False.

    Returns:
        bool: True if the process was successfully terminated, False otherwise.

    Raises:
        psutil.NoSuchProcess: If the process with the given PID doesn't exist.
        psutil.AccessDenied: If insufficient permissions to terminate the process.

    Examples:
        >>> success = kill_process(1234)
        🛑 Terminating process 1234...
        ✅ Process 1234 terminated successfully
        >>> success = kill_process(5678, force=True)
        Force killing process 5678...
        ✅ Process 5678 terminated successfully
    """
    try:
        proc = psutil.Process(pid)
        
        if force:
            print(f"Force killing process {pid}...")
            proc.kill()  # SIGKILL
        else:
            print(f"🛑 Terminating process {pid}...")
            proc.terminate()  # SIGTERM
        
        # Wait for process to die
        try:
            proc.wait(timeout=5)
            print(f"✅ Process {pid} terminated successfully")
            return True
        except psutil.TimeoutExpired:
            if not force:
                print(f"⚠️  Process {pid} didn't terminate, trying force kill...")
                return kill_process(pid, force=True)
            else:
                print(f"❌ Failed to kill process {pid}")
                return False
                
    except psutil.NoSuchProcess:
        print(f"✅ Process {pid} already terminated")
        return True
    except psutil.AccessDenied:
        print(f"❌ Access denied to kill process {pid}")
        return False
    except Exception as e:
        print(f"❌ Error killing process {pid}: {e}")
        return False
# --------------------------------------------------------------------------------- end kill_process()

# ---------------------------------------------
# --- Callable Functions from other modules ---
# ---------------------------------------------

# --------------------------------------------------------------------------------- main()
def main():
    """Main entry point for the emergency process killer utility.

    Orchestrates the process of finding and terminating stuck data processing
    operations. Provides an interactive interface for users to select which
    processes to terminate, with options to kill individual processes or all
    detected stuck processes at once.

    The function performs the following operations:
    1. Scans for potentially stuck data processing processes
    2. Displays detailed information about found processes
    3. Provides interactive options for process termination
    4. Executes the selected termination actions
    5. Provides guidance for preventing future hanging issues

    Returns:
        int: Exit code (0 for success, 1 for failure) suitable for command-line usage.

    Examples:
        >>> exit_code = main()
        Emergency Process Killer for Stuck Data Processing
        ============================================================
        Found 2 potentially stuck processes:

        1. PID 1234 - python
           CPU: 0.0%
           Command: python run_data_processing.py

        Enter process number to kill (or 'all' for all, 'q' to quit): 1
    """
    print("Emergency Process Killer for Stuck Data Processing")
    print("=" * 60)
    
    # Find potentially stuck processes
    stuck_processes = find_stuck_processes()
    
    if not stuck_processes:
        print("✅ No stuck data processing processes found")
        return 0
    
    print(f"Found {len(stuck_processes)} potentially stuck processes:")
    print()
    
    for i, proc in enumerate(stuck_processes, 1):
        print(f"{i}. PID {proc['pid']} - {proc['name']}")
        print(f"   CPU: {proc['cpu_percent']:.1f}%")
        print(f"   Command: {proc['cmdline']}")
        print()
    
    # Ask user what to do
    try:
        choice = input("Enter process number to kill (or 'all' for all, 'q' to quit): ").strip().lower()
        
        if choice == 'q':
            print("Cancelled.")
            return 0
        elif choice == 'all':
            print("Killing all stuck processes...")
            success_count = 0
            for proc in stuck_processes:
                if kill_process(proc['pid']):
                    success_count += 1
            print(f"✅ Successfully killed {success_count}/{len(stuck_processes)} processes")
        else:
            try:
                proc_index = int(choice) - 1
                if 0 <= proc_index < len(stuck_processes):
                    proc = stuck_processes[proc_index]
                    kill_process(proc['pid'])
                else:
                    print("❌ Invalid process number")
                    return 1
            except ValueError:
                print("❌ Invalid input")
                return 1
                
    except KeyboardInterrupt:
        print("\nCancelled.")
        return 0
    
    print("\nTips to prevent hanging:")
    print("1. Use the enhanced monitoring: uv run python run_data_processing_with_monitoring.py")
    print("2. Process smaller batches: set MAX_DOCS=1 in .env")
    print("3. Monitor with: uv run python monitor_process_health.py")
    
    return 0
# --------------------------------------------------------------------------------- end main()


# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It serves as the entry point for command-line execution of the emergency
# process killer utility.

if __name__ == "__main__":
    # --- Direct Execution Entry Point ---
    # Execute the main function and exit with appropriate status code
    # This allows the script to be used in emergency situations and automation
    sys.exit(main())
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================
