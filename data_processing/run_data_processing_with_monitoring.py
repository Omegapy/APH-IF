# -------------------------------------------------------------------------
# File: run_data_processing_with_monitoring.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-03-2025
# File Path: data_processing/run_data_processing_with_monitoring.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   This module provides comprehensive monitoring and orchestration for the
#   APH-IF data processing pipeline. It implements enhanced monitoring capabilities
#   including real-time API call tracking, performance metrics collection, error
#   analysis, and post-processing reporting. The module serves as the primary
#   entry point for running data processing operations with full observability
#   and diagnostic capabilities, ensuring robust execution and detailed insights
#   into system performance and behavior.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: get_neo4j_config()
# - Function: setup_monitoring()
# - Function: run_data_processing_pipeline()
# - Function: main()
# - Constants: None
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: os (environment operations), sys (system operations), argparse (command-line parsing),
#   subprocess (process management), time (timing operations), datetime (date/time handling),
#   pathlib (path operations), typing (type hints)
# - Third-Party: dotenv (environment variable loading)
# - Local Project Modules: common.api_monitor (API monitoring utilities),
#   common.monitoring_dashboard (dashboard and reporting utilities)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module is designed to be the primary entry point for running APH-IF
# data processing operations with comprehensive monitoring. It can be executed
# directly from the command line with various configuration options, or
# integrated into larger automation workflows. The module provides detailed
# monitoring, reporting, and analysis capabilities for data processing operations.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

APH-IF Data Processing Pipeline with Enhanced Monitoring

Provides comprehensive monitoring and orchestration for the APH-IF data processing
pipeline, including real-time API call tracking, performance metrics collection,
error analysis, and detailed post-processing reporting with recommendations.

Features:
- Automatic monitoring setup and configuration
- Real-time progress tracking with API statistics
- Comprehensive error handling and reporting
- Post-processing analysis and recommendations
- Monitoring data export and archival

Usage:
    # Run complete pipeline with monitoring
    uv run python run_data_processing_with_monitoring.py

    # Run with specific monitoring level
    uv run python run_data_processing_with_monitoring.py --log-level detailed

    # Run in test mode with monitoring
    uv run python run_data_processing_with_monitoring.py --test-mode

    # Run with custom monitoring output directory
    uv run python run_data_processing_with_monitoring.py --monitoring-dir custom_logs/

    # Run only initial build with monitoring
    uv run python run_data_processing_with_monitoring.py --initial-only

Environment Variables:
    VERBOSE=true                    # Enable detailed logging
    MONITORING_ENABLED=true         # Enable API monitoring (default: true)
    MONITORING_LOG_LEVEL=standard   # Monitoring detail level
    MONITORING_DIR=monitoring_logs  # Monitoring output directory

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
#!/usr/bin/env python3
import os  # Operating system interface for environment operations
import sys  # System-specific parameters and functions
import argparse  # Command-line argument parsing utilities
import subprocess  # Subprocess management for external process execution
import time  # Time-related functions for delays and timing
from datetime import datetime  # Date and time handling utilities
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Optional  # Type hinting support for optional values

# Add project root to path for imports (data_processing is one level down from root)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Third-party library imports
try:
    from dotenv import load_dotenv  # Environment variable loading from .env files
except ImportError as e:
    print(f"Error importing dotenv: {e}")
    sys.exit(1)

# Local application/library specific imports
try:
    from common.api_monitor import LogLevel, configure_monitoring  # API monitoring utilities
    from common.monitoring_dashboard import MonitoringDashboard, generate_monitoring_report  # Dashboard and reporting
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Make sure you're running from the data_processing directory with UV.")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv(project_root / '.env')


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

# --------------------------------------------------------------------------------- get_neo4j_config()
def get_neo4j_config():
    """Retrieves Neo4j database configuration from environment variables.

    Extracts the Neo4j connection parameters that were set by the set_environment.py
    module, providing a centralized way to access database configuration for
    monitoring and validation purposes.

    Returns:
        dict: A dictionary containing Neo4j connection parameters with keys:
            - 'uri' (str): Neo4j database URI
            - 'username' (str): Neo4j database username
            - 'password' (str): Neo4j database password

    Examples:
        >>> config = get_neo4j_config()
        >>> print(config['uri'])
        bolt://localhost:7687
    """
    return {
        'uri': os.getenv('NEO4J_URI'),
        'username': os.getenv('NEO4J_USERNAME'),
        'password': os.getenv('NEO4J_PASSWORD')
    }
# --------------------------------------------------------------------------------- end get_neo4j_config()

# --------------------------------------------------------------------------------- setup_monitoring()
def setup_monitoring(monitoring_dir: Path, log_level: LogLevel) -> tuple:
    """Configures and initializes comprehensive monitoring for data processing operations.

    Sets up the monitoring infrastructure including directory creation, monitor
    configuration, and initialization of both LLM API and Neo4j operation monitoring.
    Provides detailed feedback about the monitoring configuration and file locations.

    Args:
        monitoring_dir (Path): Directory path where monitoring logs will be stored.
        log_level (LogLevel): Monitoring detail level (minimal, standard, detailed, debug).

    Returns:
        tuple: A tuple containing (llm_monitor, neo4j_monitor) instances for
            tracking API calls and database operations respectively.

    Examples:
        >>> from pathlib import Path
        >>> from common.api_monitor import LogLevel
        >>> monitoring_dir = Path('monitoring_logs')
        >>> log_level = LogLevel.STANDARD
        >>> llm_monitor, neo4j_monitor = setup_monitoring(monitoring_dir, log_level)
        Monitoring configured:
           Directory: monitoring_logs
           Log Level: standard
           LLM Monitor: monitoring_logs/llm_api_calls.jsonl
           Neo4j Monitor: monitoring_logs/neo4j_operations.jsonl
    """
    monitoring_dir.mkdir(parents=True, exist_ok=True)

    # Configure global monitoring
    llm_monitor, neo4j_monitor = configure_monitoring(log_level, monitoring_dir)

    print(f"Monitoring configured:")
    print(f"   Directory: {monitoring_dir}")
    print(f"   Log Level: {log_level.value}")
    print(f"   LLM Monitor: {monitoring_dir / 'llm_api_calls.jsonl'}")
    print(f"   Neo4j Monitor: {monitoring_dir / 'neo4j_operations.jsonl'}")

    return llm_monitor, neo4j_monitor
# --------------------------------------------------------------------------------- end setup_monitoring()

# ---------------------------------------------
# --- Callable Functions from other modules ---
# ---------------------------------------------

# --------------------------------------------------------------------------------- run_data_processing_pipeline()
def run_data_processing_pipeline(args) -> int:
    """Executes the complete APH-IF data processing pipeline with comprehensive monitoring.

    Orchestrates the entire data processing workflow including monitoring setup,
    pipeline execution, real-time dashboard management, and post-processing
    analysis. Provides detailed feedback throughout the process and generates
    comprehensive reports upon completion.

    The function handles:
    - Monitoring infrastructure setup and configuration
    - Environment variable configuration for subprocess execution
    - Optional real-time dashboard management
    - Pipeline execution with real-time output
    - Post-processing analysis and reporting
    - Error handling and cleanup operations

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing
            configuration options for monitoring, pipeline execution, and
            reporting settings.

    Returns:
        int: Exit code (0 for success, non-zero for failure) suitable for
            command-line usage and automation workflows.

    Raises:
        KeyboardInterrupt: If user interrupts execution with Ctrl+C.
        Exception: For various pipeline execution or monitoring errors.

    Examples:
        >>> import argparse
        >>> args = argparse.Namespace(
        ...     monitoring_dir='monitoring_logs',
        ...     log_level='standard',
        ...     test_mode=False,
        ...     generate_report=True
        ... )
        >>> exit_code = run_data_processing_pipeline(args)
        Starting APH-IF Data Processing Pipeline with Enhanced Monitoring
        ================================================================================
        Monitoring configured:
           Directory: monitoring_logs
           Log Level: standard
        ...
    """
    print("Starting APH-IF Data Processing Pipeline with Enhanced Monitoring")
    print("=" * 80)
    
    # Setup monitoring
    monitoring_dir = Path(args.monitoring_dir)
    log_level = LogLevel(args.log_level)
    
    llm_monitor, neo4j_monitor = setup_monitoring(monitoring_dir, log_level)

    # Set environment variables for monitoring
    os.environ['MONITORING_ENABLED'] = 'true'
    os.environ['MONITORING_LOG_LEVEL'] = log_level.value
    os.environ['MONITORING_DIR'] = str(monitoring_dir)

    # Build command arguments - run from current directory (data_processing)
    cmd_args = [sys.executable, '-m', 'processing.launch_data_processing']

    if args.test_mode:
        cmd_args.append('--test-mode')
    if args.dry_run:
        cmd_args.append('--dry-run')
    if args.initial_only:
        cmd_args.append('--initial-only')
    if args.augmentation_only:
        cmd_args.append('--augmentation-only')
    if args.skip_initial:
        cmd_args.append('--skip-initial')
    if args.skip_augmentation:
        cmd_args.append('--skip-augmentation')

    # Working directory is current directory (data_processing)
    data_processing_dir = Path.cwd()

    print(f"\nPipeline Configuration:")
    print(f"   Command: {' '.join(cmd_args)}")
    print(f"   Working Directory: {data_processing_dir}")
    print(f"   Test Mode: {args.test_mode}")
    print(f"   Monitoring: {monitoring_dir}")

    # Start monitoring dashboard in background if requested
    dashboard_process = None
    if args.watch:
        print(f"\nStarting real-time monitoring dashboard...")
        dashboard_cmd = [
            sys.executable, '../monitor_api_calls.py', 'watch',
            '--input', str(monitoring_dir),
            '--refresh', str(args.watch_refresh)
        ]
        try:
            dashboard_process = subprocess.Popen(
                dashboard_cmd,
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"   Dashboard PID: {dashboard_process.pid}")
        except Exception as e:
            print(f"   Warning: Could not start dashboard: {e}")

    start_time = datetime.now()

    try:
        print(f"\nStarting Data Processing Pipeline...")
        print("=" * 80)
        
        # Run the data processing pipeline with environment variables
        env = os.environ.copy()  # Copy current environment
        env.update({
            'MONITORING_ENABLED': 'true',
            'MONITORING_LOG_LEVEL': log_level.value,
            'MONITORING_DIR': str(monitoring_dir)
        })

        result = subprocess.run(
            cmd_args,
            cwd=data_processing_dir,
            capture_output=False,  # Show output in real-time
            text=True,
            env=env  # Pass environment variables
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 80)
        print(f"Pipeline Execution Completed")
        print(f"   Duration: {duration}")
        print(f"   Exit Code: {result.returncode}")
        print("=" * 80)

        # Generate monitoring report
        if args.generate_report:
            print(f"\nGenerating Monitoring Report...")
            report_path = monitoring_dir / f"pipeline_report_{start_time.strftime('%Y%m%d_%H%M%S')}.html"

            success = generate_monitoring_report(
                monitoring_dir=monitoring_dir,
                output_path=report_path,
                format='html'
            )

            if success:
                print(f"   Report generated: {report_path}")
            else:
                print(f"   Failed to generate report")

        # Show monitoring summary
        print(f"\nMonitoring Summary:")
        try:
            dashboard = MonitoringDashboard()
            records_loaded = dashboard.load_data(monitoring_dir)

            if records_loaded > 0:
                stats = dashboard.get_realtime_stats()
                print(f"   Total API Calls: {stats['total_records']:,}")
                print(f"   LLM Calls: {stats['llm_calls']:,}")
                print(f"   Neo4j Operations: {stats['neo4j_operations']:,}")
                print(f"   Success Rate: {stats['success_rate']:.1%}")
                print(f"   Avg Duration: {stats['avg_duration_ms']:.1f}ms")

                # Generate quick analysis
                report = dashboard.generate_comprehensive_report()
                if report.error_analysis['total_errors'] > 0:
                    print(f"   Errors Detected: {report.error_analysis['total_errors']}")

                if report.recommendations:
                    print(f"\nKey Recommendations:")
                    for i, rec in enumerate(report.recommendations[:3], 1):
                        print(f"   {i}. {rec}")
            else:
                print("   No monitoring data collected")

        except Exception as e:
            print(f"   Error analyzing monitoring data: {e}")

        return result.returncode

    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 130

    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        return 1

    finally:
        # Stop dashboard process if running
        if dashboard_process:
            try:
                dashboard_process.terminate()
                dashboard_process.wait(timeout=5)
                print(f"\nMonitoring dashboard stopped")
            except Exception as e:
                print(f"Warning: Could not stop dashboard: {e}")
# --------------------------------------------------------------------------------- end run_data_processing_pipeline()

# --------------------------------------------------------------------------------- main()
def main():
    """Main entry point for the APH-IF data processing pipeline with monitoring.

    Provides comprehensive command-line interface for running data processing
    operations with full monitoring capabilities. Parses command-line arguments,
    validates environment configuration, and orchestrates the complete pipeline
    execution with monitoring and reporting.

    The function handles:
    - Command-line argument parsing and validation
    - Environment configuration validation
    - Pipeline execution orchestration
    - Error handling and user feedback

    Returns:
        int: Exit code (0 for success, 1 for failure) suitable for command-line
            usage and automation workflows.

    Examples:
        >>> exit_code = main()
        APH-IF Data Processing with Enhanced Monitoring
        Starting APH-IF Data Processing Pipeline with Enhanced Monitoring
        ================================================================================
        ...
    """
    parser = argparse.ArgumentParser(
        description="APH-IF Data Processing with Enhanced Monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Pipeline control options
    parser.add_argument('--test-mode', action='store_true',
                       help='Run in test mode with test database')
    parser.add_argument('--dry-run', action='store_true',
                       help='Preview changes without writing to database')
    parser.add_argument('--initial-only', action='store_true',
                       help='Run only initial graph build')
    parser.add_argument('--augmentation-only', action='store_true',
                       help='Run only relationship augmentation')
    parser.add_argument('--skip-initial', action='store_true',
                       help='Skip initial graph build')
    parser.add_argument('--skip-augmentation', action='store_true',
                       help='Skip relationship augmentation')
    
    # Monitoring options
    parser.add_argument('--monitoring-dir', default='monitoring_logs',
                       help='Monitoring output directory (default: monitoring_logs)')
    parser.add_argument('--log-level', choices=['minimal', 'standard', 'detailed', 'debug'],
                       default='standard', help='Monitoring detail level (default: standard)')
    parser.add_argument('--generate-report', action='store_true', default=True,
                       help='Generate monitoring report after completion (default: true)')
    parser.add_argument('--no-report', dest='generate_report', action='store_false',
                       help='Skip monitoring report generation')
    
    # Real-time monitoring options
    parser.add_argument('--watch', action='store_true',
                       help='Start real-time monitoring dashboard')
    parser.add_argument('--watch-refresh', type=int, default=5,
                       help='Dashboard refresh interval in seconds (default: 5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.initial_only and args.augmentation_only:
        print("Error: Cannot specify both --initial-only and --augmentation-only")
        return 1
    
    if args.skip_initial and args.skip_augmentation:
        print("Error: Cannot skip both initial build and augmentation")
        return 1
    
    # Check environment
    try:
        config = get_neo4j_config()
        if not all([config['uri'], config['username'], config['password']]):
            print("Neo4j configuration incomplete. Please check your environment.")
            return 1

        if not os.getenv('OPENAI_API_KEY'):
            print("OPENAI_API_KEY not found. Please set your OpenAI API key.")
            return 1
            
    except Exception as e:
        print(f"Environment validation failed: {e}")
        return 1
    
    # Run the pipeline
    return run_data_processing_pipeline(args)
# --------------------------------------------------------------------------------- end main()


# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It serves as the entry point for command-line execution of the APH-IF
# data processing pipeline with comprehensive monitoring capabilities.

if __name__ == "__main__":
    # --- Direct Execution Entry Point ---
    # Execute the main function and exit with appropriate status code
    # This allows the script to be used in automation workflows and monitoring systems
    sys.exit(main())
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================
