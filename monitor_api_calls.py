# -------------------------------------------------------------------------
# File: monitor_api_calls.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-03-2025
# File Path: monitor_api_calls.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   This module provides comprehensive command-line utilities for monitoring and
#   analyzing API calls during APH-IF data processing operations. It offers
#   real-time monitoring capabilities, detailed report generation, statistical
#   analysis, and troubleshooting tools for API performance and error tracking.
#   The module serves as the primary interface for observing and analyzing
#   system behavior during data processing workflows, enabling performance
#   optimization and issue diagnosis.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: cmd_report()
# - Function: cmd_stats()
# - Function: cmd_watch()
# - Function: cmd_analyze()
# - Function: cmd_clear()
# - Function: main()
# - Constants: None
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: argparse (command-line parsing), sys (system operations), time (timing operations),
#   datetime (date/time handling), pathlib (path operations), typing (type hints)
# - Third-Party: None
# - Local Project Modules: common.monitoring_dashboard (dashboard and reporting utilities),
#   common.api_monitor (API monitoring core functionality)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module is designed to be run as a command-line utility for monitoring
# and analyzing API calls during data processing operations. It provides multiple
# subcommands for different monitoring tasks including real-time watching,
# report generation, statistical analysis, and log management. The module
# integrates with the APH-IF monitoring infrastructure to provide comprehensive
# observability into system performance and behavior.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

APH-IF API Monitoring Command-Line Utility

Provides comprehensive command-line utilities for monitoring and analyzing API
calls during data processing operations, including real-time monitoring, report
generation, statistical analysis, and troubleshooting capabilities for performance
optimization and issue diagnosis.

Usage:
    # Generate monitoring report
    python monitor_api_calls.py report --input monitoring_logs/ --output report.html

    # Real-time monitoring
    python monitor_api_calls.py watch --input monitoring_logs/

    # Show current statistics
    python monitor_api_calls.py stats --input monitoring_logs/

    # Analyze specific time period
    python monitor_api_calls.py report --input monitoring_logs/ --since "2024-01-01 10:00" --until "2024-01-01 12:00"

Examples:
    # Monitor data processing pipeline
    python monitor_api_calls.py watch --input monitoring_logs/ --refresh 5

    # Generate comprehensive HTML report
    python monitor_api_calls.py report --input monitoring_logs/ --output monitoring_report.html --format html

    # Export CSV for analysis
    python monitor_api_calls.py report --input monitoring_logs/ --output data.csv --format csv

    # Check for errors and performance issues
    python monitor_api_calls.py analyze --input monitoring_logs/ --check-errors --check-performance

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
#!/usr/bin/env python3
import argparse  # Command-line argument parsing utilities
import sys  # System-specific parameters and functions
import time  # Time-related functions for delays and timing
from datetime import datetime, timedelta  # Date and time handling utilities
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Optional  # Type hinting support for optional values

# Third-party library imports
# None - this utility uses only standard library and local modules

# Local application/library specific imports
try:
    from common.monitoring_dashboard import MonitoringDashboard, generate_monitoring_report  # Dashboard and reporting
    from common.api_monitor import LogLevel, get_monitoring_summary  # API monitoring core functionality
except ImportError as e:
    print(f"Error: Could not import monitoring modules: {e}")
    print("Make sure you're running from the project root directory.")
    sys.exit(1)


# =========================================================================
# Global Constants / Variables
# =========================================================================
# No global constants defined in this module


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# These are functions that are not methods of any specific class within this module.

# ---------------------------------------------
# --- Command Handler Functions ---
# ---------------------------------------------

# --------------------------------------------------------------------------------- cmd_report()
def cmd_report(args):
    """Generates comprehensive monitoring reports from API call data.

    Creates detailed reports from monitoring data in various formats including
    HTML, JSON, and CSV. Supports time-based filtering and comprehensive
    analysis of API performance, error rates, and usage patterns.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - input: Input directory or file path for monitoring data
            - output: Output file path for the generated report
            - format: Output format ('html', 'json', or 'csv')

    Returns:
        int: Exit code (0 for success, 1 for failure).

    Examples:
        >>> args = argparse.Namespace(input='monitoring_logs/', output='report.html', format='html')
        >>> exit_code = cmd_report(args)
        Generating monitoring report from monitoring_logs/...
        ✅ Report generated successfully: report.html
    """
    print(f"Generating monitoring report from {args.input}...")

    success = generate_monitoring_report(
        monitoring_dir=args.input,
        output_path=args.output,
        format=args.format
    )

    if success:
        print(f"✅ Report generated successfully: {args.output}")
        return 0
    else:
        print("❌ Failed to generate report")
        return 1
# --------------------------------------------------------------------------------- end cmd_report()

# --------------------------------------------------------------------------------- cmd_stats()
def cmd_stats(args):
    """Displays current monitoring statistics and performance metrics.

    Loads monitoring data and presents a comprehensive statistical summary
    including total API calls, recent activity, success rates, performance
    metrics, and timing information. Provides a quick overview of system
    performance and activity levels.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - input: Input directory or file path for monitoring data

    Returns:
        int: Exit code (0 for success, 1 if no data found).

    Examples:
        >>> args = argparse.Namespace(input='monitoring_logs/')
        >>> exit_code = cmd_stats(args)
        Monitoring Statistics (1,234 records loaded)
        ============================================================
        Total API Calls: 1,234
        Recent Hour: 45
        LLM Calls: 890
        Neo4j Operations: 344
        Success Rate: 98.5%
        Avg Duration: 1,250.3ms
        Last Updated: 2025-01-15 14:30:25
    """
    dashboard = MonitoringDashboard()
    records_loaded = dashboard.load_data(args.input)

    if records_loaded == 0:
        print(f"No monitoring data found in {args.input}")
        return 1

    print(f"Monitoring Statistics ({records_loaded:,} records loaded)")
    print("=" * 60)

    stats = dashboard.get_realtime_stats()

    print(f"Total API Calls: {stats['total_records']:,}")
    print(f"Recent Hour: {stats['recent_hour_records']:,}")
    print(f"LLM Calls: {stats['llm_calls']:,}")
    print(f"Neo4j Operations: {stats['neo4j_operations']:,}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Avg Duration: {stats['avg_duration_ms']:.1f}ms")
    print(f"Last Updated: {stats['last_updated']}")

    return 0
# --------------------------------------------------------------------------------- end cmd_stats()

# --------------------------------------------------------------------------------- cmd_watch()
def cmd_watch(args):
    """Provides real-time monitoring of API calls with continuous updates.

    Implements a real-time monitoring mode that continuously watches for new
    monitoring data and displays live statistics with configurable refresh
    intervals. Shows key metrics including total calls, success rates, average
    duration, and recent activity in a compact, continuously updating format.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - input: Input directory or file path for monitoring data
            - refresh: Refresh interval in seconds for updates

    Returns:
        int: Exit code (0 for success, typically after Ctrl+C interruption).

    Examples:
        >>> args = argparse.Namespace(input='monitoring_logs/', refresh=5)
        >>> exit_code = cmd_watch(args)
        Watching monitoring_logs/ for API monitoring data...
        Refresh interval: 5 seconds
        Press Ctrl+C to stop
        ============================================================
        [14:30:25] Calls: 1,234 | Success: 98.5% | Avg: 1250ms | Recent: 45
    """
    print(f"Watching {args.input} for API monitoring data...")
    print(f"Refresh interval: {args.refresh} seconds")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        while True:
            dashboard = MonitoringDashboard()
            records_loaded = dashboard.load_data(args.input)

            if records_loaded > 0:
                stats = dashboard.get_realtime_stats()

                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\r[{timestamp}] Calls: {stats['total_records']:,} | "
                      f"Success: {stats['success_rate']:.1%} | "
                      f"Avg: {stats['avg_duration_ms']:.0f}ms | "
                      f"Recent: {stats['recent_hour_records']:,}", end="", flush=True)
            else:
                timestamp = datetime.now().strftime("%H:%M:%S")
                print(f"\r[{timestamp}] No monitoring data found...", end="", flush=True)

            time.sleep(args.refresh)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")
        return 0
# --------------------------------------------------------------------------------- end cmd_watch()

# --------------------------------------------------------------------------------- cmd_analyze()
def cmd_analyze(args):
    """Performs comprehensive analysis of monitoring data for issues and performance problems.

    Analyzes monitoring data to identify errors, performance issues, and system
    problems. Provides detailed error analysis including error types and rates,
    performance metrics analysis with percentile calculations, and actionable
    recommendations for system optimization and issue resolution.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - input: Input directory or file path for monitoring data
            - check_errors: Flag to enable error analysis
            - check_performance: Flag to enable performance analysis

    Returns:
        int: Exit code (0 for success, 1 if no data found).

    Examples:
        >>> args = argparse.Namespace(input='monitoring_logs/', check_errors=True, check_performance=True)
        >>> exit_code = cmd_analyze(args)
        Analyzing 1,234 monitoring records...
        ============================================================
        ⚠️  ERRORS DETECTED:
           Total Errors: 15
           Error Rate: 1.2%
           Error Types:
             - TimeoutError: 8
             - ConnectionError: 7

        PERFORMANCE ANALYSIS:
           Avg Duration: 1,250.3ms
           P95 Duration: 3,450.0ms
           P99 Duration: 8,920.0ms
           Slow Calls (>5s): 12
           Very Slow Calls (>30s): 2
    """
    dashboard = MonitoringDashboard()
    records_loaded = dashboard.load_data(args.input)

    if records_loaded == 0:
        print(f"No monitoring data found in {args.input}")
        return 1

    print(f"Analyzing {records_loaded:,} monitoring records...")
    print("=" * 60)

    report = dashboard.generate_comprehensive_report()

    # Check for errors
    if args.check_errors:
        error_analysis = report.error_analysis
        if error_analysis['total_errors'] > 0:
            print(f"⚠️  ERRORS DETECTED:")
            print(f"   Total Errors: {error_analysis['total_errors']}")
            print(f"   Error Rate: {error_analysis['error_rate']:.1%}")

            if error_analysis.get('error_types'):
                print("   Error Types:")
                for error_type, count in error_analysis['error_types'].items():
                    print(f"     - {error_type}: {count}")
        else:
            print("✅ No errors detected")

    # Check performance
    if args.check_performance:
        perf = report.performance_metrics
        print(f"\nPERFORMANCE ANALYSIS:")
        print(f"   Avg Duration: {perf.get('avg_duration_ms', 0):.1f}ms")
        print(f"   P95 Duration: {perf.get('p95_duration_ms', 0):.1f}ms")
        print(f"   P99 Duration: {perf.get('p99_duration_ms', 0):.1f}ms")
        print(f"   Slow Calls (>5s): {perf.get('slow_calls', 0)}")
        print(f"   Very Slow Calls (>30s): {perf.get('very_slow_calls', 0)}")

        if perf.get('avg_duration_ms', 0) > 2000:
            print("   ⚠️  Average response time is high")
        if perf.get('slow_calls', 0) > 0:
            print(f"   ⚠️  {perf.get('slow_calls', 0)} slow calls detected")

    # Show recommendations
    print(f"\nRECOMMENDATIONS:")
    for i, rec in enumerate(report.recommendations, 1):
        print(f"   {i}. {rec}")

    return 0
# --------------------------------------------------------------------------------- end cmd_analyze()

# --------------------------------------------------------------------------------- cmd_clear()
def cmd_clear(args):
    """Clears monitoring log files from the specified directory.

    Removes all monitoring log files (*.jsonl and *.json) from the specified
    directory. Provides safety confirmation unless force flag is used, and
    gives detailed feedback about the deletion process including file counts
    and any errors encountered.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - input: Input directory path containing monitoring logs to clear
            - force: Flag to skip confirmation prompt

    Returns:
        int: Exit code (0 for success, 1 if directory doesn't exist).

    Examples:
        >>> args = argparse.Namespace(input='monitoring_logs/', force=False)
        >>> exit_code = cmd_clear(args)
        Are you sure you want to clear all monitoring logs in monitoring_logs/? (y/N): y
        Deleted: monitoring_logs/llm_api_calls.jsonl
        Deleted: monitoring_logs/neo4j_operations.jsonl
        ✅ Cleared 2 monitoring log files
    """
    monitoring_dir = Path(args.input)

    if not monitoring_dir.exists():
        print(f"Directory {monitoring_dir} does not exist")
        return 1

    if not args.force:
        response = input(f"Are you sure you want to clear all monitoring logs in {monitoring_dir}? (y/N): ")
        if response.lower() != 'y':
            print("Operation cancelled")
            return 0

    log_files = list(monitoring_dir.glob("*.jsonl")) + list(monitoring_dir.glob("*.json"))

    if not log_files:
        print("No monitoring log files found")
        return 0

    for file_path in log_files:
        try:
            file_path.unlink()
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    print(f"✅ Cleared {len(log_files)} monitoring log files")
    return 0
# --------------------------------------------------------------------------------- end cmd_clear()

# ---------------------------------------------
# --- Callable Functions from other modules ---
# ---------------------------------------------

# --------------------------------------------------------------------------------- main()
def main():
    """Main entry point for the API monitoring command-line utility.

    Provides comprehensive command-line interface for API monitoring operations
    including report generation, real-time monitoring, statistical analysis,
    issue analysis, and log management. Parses command-line arguments and
    dispatches to appropriate command handler functions.

    Returns:
        int: Exit code (0 for success, 1 for failure) suitable for command-line
            usage and automation workflows.

    Examples:
        >>> exit_code = main()  # With 'report' subcommand
        Generating monitoring report from monitoring_logs/...
        ✅ Report generated successfully: monitoring_report.html
        >>> exit_code = main()  # With 'watch' subcommand
        Watching monitoring_logs/ for API monitoring data...
        Refresh interval: 5 seconds
        Press Ctrl+C to stop
        ============================================================
        [14:30:25] Calls: 1,234 | Success: 98.5% | Avg: 1250ms | Recent: 45
    """
    parser = argparse.ArgumentParser(
        description="APH-IF API Monitoring Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate monitoring report')
    report_parser.add_argument('--input', '-i', default='monitoring_logs/', 
                              help='Input directory or file (default: monitoring_logs/)')
    report_parser.add_argument('--output', '-o', default='monitoring_report.html',
                              help='Output file path (default: monitoring_report.html)')
    report_parser.add_argument('--format', '-f', choices=['html', 'json', 'csv'], default='html',
                              help='Output format (default: html)')
    report_parser.add_argument('--since', help='Start time (YYYY-MM-DD HH:MM)')
    report_parser.add_argument('--until', help='End time (YYYY-MM-DD HH:MM)')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show current statistics')
    stats_parser.add_argument('--input', '-i', default='monitoring_logs/',
                             help='Input directory or file (default: monitoring_logs/)')
    
    # Watch command
    watch_parser = subparsers.add_parser('watch', help='Real-time monitoring')
    watch_parser.add_argument('--input', '-i', default='monitoring_logs/',
                             help='Input directory or file (default: monitoring_logs/)')
    watch_parser.add_argument('--refresh', '-r', type=int, default=5,
                             help='Refresh interval in seconds (default: 5)')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze for issues')
    analyze_parser.add_argument('--input', '-i', default='monitoring_logs/',
                               help='Input directory or file (default: monitoring_logs/)')
    analyze_parser.add_argument('--check-errors', action='store_true',
                               help='Check for errors')
    analyze_parser.add_argument('--check-performance', action='store_true',
                               help='Check performance issues')
    
    # Clear command
    clear_parser = subparsers.add_parser('clear', help='Clear monitoring logs')
    clear_parser.add_argument('--input', '-i', default='monitoring_logs/',
                             help='Input directory (default: monitoring_logs/)')
    clear_parser.add_argument('--force', action='store_true',
                             help='Force clear without confirmation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute command
    if args.command == 'report':
        return cmd_report(args)
    elif args.command == 'stats':
        return cmd_stats(args)
    elif args.command == 'watch':
        return cmd_watch(args)
    elif args.command == 'analyze':
        return cmd_analyze(args)
    elif args.command == 'clear':
        return cmd_clear(args)
    else:
        print(f"Unknown command: {args.command}")
        return 1
# --------------------------------------------------------------------------------- end main()


# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It serves as the entry point for command-line execution of the API monitoring
# utility with comprehensive subcommand support.

if __name__ == "__main__":
    # --- Direct Execution Entry Point ---
    # Execute the main function and exit with appropriate status code
    # This allows the script to be used in monitoring workflows and automation
    sys.exit(main())
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================
