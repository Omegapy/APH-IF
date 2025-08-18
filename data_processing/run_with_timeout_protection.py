# -------------------------------------------------------------------------
# File: run_with_timeout_protection.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-03-2025
# File Path: data_processing/run_with_timeout_protection.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   This module provides comprehensive timeout protection and process monitoring
#   for APH-IF data processing operations. It implements multiple layers of
#   protection against hanging processes, including process timeout monitoring,
#   connection pool management, automatic restart on hang detection, and enhanced
#   error handling. The module serves as a robust wrapper around data processing
#   scripts to ensure system stability and prevent resource exhaustion from
#   unresponsive operations.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class: TimeoutProtection
# - Function: setup_environment_for_stability()
# - Function: main()
# - Constants: None
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os (environment operations), sys (system operations), time (timing operations),
#   signal (process signaling), subprocess (process management), threading (concurrent execution),
#   datetime (date/time handling), pathlib (path operations)
# - Third-Party: psutil (process utilities, optional fallback)
# - Local Project Modules: None (standalone wrapper utility)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module is designed to be run as a wrapper script around data processing
# operations that may be prone to hanging. It can be executed directly from the
# command line with various options to control timeout behavior and monitoring.
# The module provides a safety layer for long-running data processing tasks,
# ensuring they don't consume system resources indefinitely.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

Timeout Protection Wrapper for APH-IF Data Processing

Provides comprehensive timeout protection and process monitoring for data
processing operations, implementing multiple layers of protection against
hanging processes and ensuring system stability through robust error handling
and automatic process termination.

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
#!/usr/bin/env python3
import os  # Operating system interface for environment operations
import sys  # System-specific parameters and functions
import time  # Time-related functions for delays and timing
import signal  # Signal handling for process control
import subprocess  # Subprocess management for external process execution
import threading  # Thread-based parallelism for concurrent monitoring
from datetime import datetime, timedelta  # Date and time handling utilities
from pathlib import Path  # Object-oriented filesystem path handling

# Add project root to path for module imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party library imports
# psutil - imported conditionally within functions for process management

# Local application/library specific imports
# None - this is a standalone wrapper utility


# =========================================================================
# Global Constants / Variables
# =========================================================================
# No global constants defined in this module


# =========================================================================
# Class Definitions
# =========================================================================

# ------------------------------------------------------------------------- TimeoutProtection
class TimeoutProtection:
    """Comprehensive timeout protection system for data processing operations.

    Provides multiple layers of protection against hanging processes, including
    total execution timeout monitoring, output timeout detection, and graceful
    process termination with fallback to force killing. The class manages
    subprocess execution with real-time output monitoring and automatic
    termination of unresponsive processes.

    The protection system monitors both total execution time and output activity
    to detect different types of hanging conditions and responds appropriately
    to maintain system stability.

    Class Attributes:
        None

    Instance Attributes:
        timeout_minutes (int): Maximum allowed execution time in minutes.
        process (subprocess.Popen): The managed subprocess instance.
        start_time (datetime): Timestamp when process execution began.
        last_output_time (datetime): Timestamp of the last output from process.
        monitoring (bool): Flag indicating if monitoring thread is active.

    Methods:
        run_with_protection(): Execute command with comprehensive timeout protection.
        _monitor_process(): Internal method for continuous process monitoring.
        _kill_process(): Internal method for process termination and cleanup.
    """
    # ----------------------
    # --- Class Variable ---
    # ----------------------
    # No class-level variables defined

    # ---------------------------------------------------------------------------------

    # -------------------
    # --- Constructor ---
    # -------------------

    # --------------------------------------------------------------------------------- __init__()
    def __init__(self, timeout_minutes=30):
        """Initializes the TimeoutProtection system with specified timeout settings.

        Sets up the timeout protection system with configurable timeout duration
        and initializes all monitoring state variables to their default values.
        The system is ready to manage subprocess execution with timeout protection
        after initialization.

        Args:
            timeout_minutes (int, optional): Maximum allowed execution time in minutes.
                Defaults to 30 minutes.
        """
        self.timeout_minutes = timeout_minutes
        self.process = None
        self.start_time = None
        self.last_output_time = None
        self.monitoring = False
    # --------------------------------------------------------------------------------- end __init__()

    # ---------------------------
    # --- Setters / Mutators ---
    # ---------------------------

    # --------------------------------------------------------------------------------- run_with_protection()
    def run_with_protection(self, cmd_args, cwd=None):
        """Executes a command with comprehensive timeout and monitoring protection.

        Runs the specified command with multiple layers of protection against hanging,
        including total execution timeout, output timeout detection, and graceful
        process termination. Provides real-time output monitoring and detailed
        execution feedback.

        The method starts a background monitoring thread that continuously checks
        for timeout conditions and automatically terminates unresponsive processes.
        It handles various error conditions and provides appropriate cleanup.

        Args:
            cmd_args (list): List of command arguments to execute (e.g., ['python', 'script.py']).
            cwd (str, optional): Working directory for command execution. Defaults to current directory.

        Returns:
            int: Process exit code (0 for success, non-zero for failure or timeout).

        Raises:
            KeyboardInterrupt: If user interrupts execution with Ctrl+C.
            Exception: For various subprocess execution errors.

        Examples:
            >>> protection = TimeoutProtection(timeout_minutes=60)
            >>> exit_code = protection.run_with_protection(['python', 'data_processing.py'])
            Starting process with 60-minute timeout protection
            Command: python data_processing.py
            Working directory: /current/directory
            ...
            Process completed in 0:45:30
            Exit code: 0
        """
        print(f"Starting process with {self.timeout_minutes}-minute timeout protection")
        print(f"Command: {' '.join(cmd_args)}")
        print(f"Working directory: {cwd or os.getcwd()}")
        print()
        
        self.start_time = datetime.now()
        self.last_output_time = self.start_time
        self.monitoring = True
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_process, daemon=True)
        monitor_thread.start()
        
        try:
            # Start the process
            self.process = subprocess.Popen(
                cmd_args,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Read output line by line
            while True:
                output = self.process.stdout.readline()
                if output == '' and self.process.poll() is not None:
                    break
                if output:
                    print(output.strip())
                    self.last_output_time = datetime.now()
            
            # Wait for process to complete
            return_code = self.process.wait()
            self.monitoring = False
            
            duration = datetime.now() - self.start_time
            print(f"\nProcess completed in {duration}")
            print(f"Exit code: {return_code}")
            
            return return_code
            
        except KeyboardInterrupt:
            print("\n⚠️  Keyboard interrupt received")
            self._kill_process()
            return 130
        except Exception as e:
            print(f"\n❌ Process error: {e}")
            self._kill_process()
            return 1
        finally:
            self.monitoring = False
    # --------------------------------------------------------------------------------- end run_with_protection()

    # -----------------------------------------------------------------------
    # --- Internal/Private Methods (Single leading underscore convention) ---
    # -----------------------------------------------------------------------

    # --------------------------------------------------------------------------------- _monitor_process()
    def _monitor_process(self):
        """Continuously monitors the managed process for timeout and hang conditions.

        This internal method runs in a separate thread to monitor the subprocess
        for various timeout conditions, including total execution timeout and
        output timeout. It automatically terminates the process if timeout
        conditions are detected.

        The monitoring includes:
        - Total execution time limit checking
        - Output activity monitoring (detects silent hangs)
        - Automatic process termination on timeout detection

        This method is not intended for direct external use.

        Examples:
            Internal usage only - called automatically by run_with_protection()
        """
        while self.monitoring and self.process:
            time.sleep(30)  # Check every 30 seconds
            
            if not self.monitoring:
                break
                
            now = datetime.now()
            
            # Check total timeout
            total_duration = now - self.start_time
            if total_duration > timedelta(minutes=self.timeout_minutes):
                print(f"\n❌ TIMEOUT: Process exceeded {self.timeout_minutes} minutes")
                self._kill_process()
                break
            
            # Check output timeout (no output for 10 minutes)
            output_duration = now - self.last_output_time
            if output_duration > timedelta(minutes=10):
                print(f"\n⚠️  WARNING: No output for {output_duration}")
                print("Process may be hanging...")
                
                # If no output for 15 minutes, kill it
                if output_duration > timedelta(minutes=15):
                    print(f"\nHANG DETECTED: No output for {output_duration}")
                    print("Killing hung process...")
                    self._kill_process()
                    break
    # --------------------------------------------------------------------------------- end _monitor_process()

    # --------------------------------------------------------------------------------- _kill_process()
    def _kill_process(self):
        """Terminates the managed process and all its child processes.

        Performs comprehensive process cleanup by terminating both the main
        process and all its child processes. Uses graceful termination (SIGTERM)
        first, followed by force killing (SIGKILL) if necessary. Handles various
        error conditions that may occur during process termination.

        The method attempts to use psutil for comprehensive process tree
        management, with fallback to basic subprocess termination if psutil
        is not available.

        This method is not intended for direct external use.

        Examples:
            Internal usage only - called automatically when timeout is detected
        """
        if not self.process:
            return
            
        try:
            import psutil
            
            # Get process and all children
            parent = psutil.Process(self.process.pid)
            children = parent.children(recursive=True)
            
            # Terminate children first
            for child in children:
                try:
                    print(f"🛑 Terminating child process {child.pid}")
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Terminate parent
            print(f"🛑 Terminating main process {parent.pid}")
            parent.terminate()
            
            # Wait for termination
            gone, alive = psutil.wait_procs(children + [parent], timeout=5)
            
            # Force kill any remaining processes
            for proc in alive:
                try:
                    print(f"Force killing process {proc.pid}")
                    proc.kill()
                except psutil.NoSuchProcess:
                    pass
                    
        except ImportError:
            # Fallback without psutil
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        except Exception as e:
            print(f"Error killing process: {e}")
    # --------------------------------------------------------------------------------- end _kill_process()

# ------------------------------------------------------------------------- end TimeoutProtection


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# These are functions that are not methods of any specific class within this module.

# --------------------------
# --- Utility Functions ---
# --------------------------

# --------------------------------------------------------------------------------- setup_environment_for_stability()
def setup_environment_for_stability():
    """Configures environment variables to improve data processing stability.

    Sets various environment variables that help prevent common stability issues
    in data processing operations, including HTTP timeouts, connection pool
    limits, and synchronous operation modes. These settings help reduce the
    likelihood of hanging processes and connection exhaustion.

    The function configures:
    - HTTP and API timeout settings
    - Connection pool size limits
    - Synchronous operation modes for stability

    Examples:
        >>> setup_environment_for_stability()
        Environment configured for stability:
           - HTTP timeout: 120s
           - Connection pool: 10 connections
           - Async mode: disabled
    """
    # HTTP connection settings
    os.environ['HTTPX_TIMEOUT'] = '120'
    os.environ['OPENAI_TIMEOUT'] = '120'
    
    # Reduce connection pool size to prevent exhaustion
    os.environ['HTTPX_POOL_CONNECTIONS'] = '10'
    os.environ['HTTPX_POOL_MAXSIZE'] = '10'
    
    # Force synchronous mode for stability
    os.environ['LANGCHAIN_ASYNC'] = 'false'
    
    print("Environment configured for stability:")
    print("   - HTTP timeout: 120s")
    print("   - Connection pool: 10 connections")
    print("   - Async mode: disabled")
    print()
# --------------------------------------------------------------------------------- end setup_environment_for_stability()

# ---------------------------------------------
# --- Callable Functions from other modules ---
# ---------------------------------------------

# --------------------------------------------------------------------------------- main()
def main():
    """Main entry point for the timeout protection wrapper utility.

    Provides command-line interface for running data processing operations with
    comprehensive timeout protection. Parses command-line arguments, configures
    the environment for stability, and executes the data processing pipeline
    with the specified timeout and monitoring settings.

    The function supports various command-line options for controlling timeout
    behavior, test modes, and monitoring configurations. It provides detailed
    feedback about the execution process and handles error conditions gracefully.

    Returns:
        int: Exit code (0 for success, non-zero for failure) suitable for command-line usage.

    Examples:
        >>> exit_code = main()
        Data Processing with Timeout Protection
        ============================================================
        Environment configured for stability:
           - HTTP timeout: 120s
           - Connection pool: 10 connections
           - Async mode: disabled

        Starting process with 60-minute timeout protection
        Command: python run_data_processing_with_monitoring.py
        ...
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run data processing with timeout protection")
    parser.add_argument('--timeout', type=int, default=60, 
                       help='Timeout in minutes (default: 60)')
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode')
    parser.add_argument('--initial-only', action='store_true',
                       help='Run only initial build')
    parser.add_argument('--monitoring-dir', default='monitoring_logs',
                       help='Monitoring directory')
    parser.add_argument('--log-level', default='detailed',
                       help='Log level')
    
    args = parser.parse_args()
    
    print("Data Processing with Timeout Protection")
    print("=" * 60)
    
    # Setup stable environment
    setup_environment_for_stability()
    
    # Build command
    cmd_args = [
        sys.executable, 
        'run_data_processing_with_monitoring.py',
        '--monitoring-dir', args.monitoring_dir,
        '--log-level', args.log_level
    ]
    
    if args.test_mode:
        cmd_args.append('--test-mode')
    if args.initial_only:
        cmd_args.append('--initial-only')
    
    # Run with protection
    protection = TimeoutProtection(timeout_minutes=args.timeout)
    return_code = protection.run_with_protection(cmd_args, cwd=Path.cwd())
    
    if return_code != 0:
        print(f"\n❌ Process failed with code {return_code}")
        print("To kill any remaining stuck processes, run:")
        print("   uv run python kill_stuck_process.py")
    
    return return_code
# --------------------------------------------------------------------------------- end main()


# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It serves as the entry point for command-line execution of the timeout
# protection wrapper utility.

if __name__ == "__main__":
    # --- Direct Execution Entry Point ---
    # Execute the main function and exit with appropriate status code
    # This allows the script to be used as a robust wrapper for data processing
    sys.exit(main())
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================
