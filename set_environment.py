# -------------------------------------------------------------------------
# File: set_environment.py
# Project: APH-IF
# Author: Alexander Ricciardi
# Date: 08-03-2025
# File Path: set_environment.py
# ------------------------------------------------------------------------

# --- Module Objective ---
#   This module provides standalone environment configuration management for the
#   APH-IF project. It handles application environment settings and Neo4j database
#   selection by directly updating the project's .env file without relying on
#   external environment managers. The module supports development, production,
#   and test database configurations with flexible override capabilities and
#   comprehensive validation to ensure proper environment setup across different
#   deployment scenarios.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Function: _load_env()
# - Function: _set_env_var()
# - Function: _get()
# - Function: _select_credentials()
# - Function: apply_settings()
# - Function: print_status()
# - Function: parse_args()
# - Function: main()
# - Constants: PROJECT_ROOT, DOTENV_PATH
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library: argparse (command-line parsing), os (environment operations),
#   pathlib (path handling), typing (type hints)
# - Third-Party: dotenv (environment variable management)
# - Local Project Modules: None (standalone environment configuration utility)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module is designed to be run as a standalone command-line utility for
# configuring the APH-IF environment. It can be executed directly to set
# application modes, database configurations, and logging levels. Other modules
# in the project rely on the environment variables set by this utility for
# proper configuration and database connectivity.

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

APH-IF Environment Configuration Manager

Provides standalone environment configuration management for the APH-IF project,
handling application environment settings and Neo4j database selection through
direct .env file manipulation with comprehensive validation and flexible
override capabilities.

Purpose
-------
Set application environment and Neo4j database selection by updating the
project's .env file directly, without relying on env_manager.py.

How it works
------------
1) Loads `.env` from the project root.
2) Applies requested mode and test DB toggle:
   - If --force-test-db=true → use TEST credentials (overrides mode)
   - Else use DEV or PROD credentials depending on --mode
3) Updates `APP_ENV`, `FORCE_TEST_DB`, `VERBOSE`, and runtime `NEO4J_URI/_USERNAME/_PASSWORD`.
4) Prints current status for verification.

Usage
-----
  python set_environment.py --mode development --force-test-db false --verbose true
  python set_environment.py --mode production   --force-test-db false
  python set_environment.py --force-test-db true                 # test DB
  python set_environment.py --preview                            # print only

Notes
-----
- This script modifies the shared .env at the repo root.
- "Test DB" overrides mode when enabled.

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
#!/usr/bin/env python3
from __future__ import annotations

import argparse  # Command-line argument parsing utilities
import os  # Operating system interface for environment operations
from pathlib import Path  # Object-oriented filesystem path handling
from typing import Tuple  # Type hinting support for tuple types

# Third-party library imports
try:
    from dotenv import load_dotenv, set_key  # Environment variable file management
except Exception as exc:  # pragma: no cover
    raise SystemExit("python-dotenv is required. Install with: uv add python-dotenv (per service)") from exc

# Local application/library specific imports
# None - this is a standalone environment configuration utility


# =========================================================================
# Global Constants / Variables
# =========================================================================
PROJECT_ROOT = Path(__file__).parent  # Project root directory path
DOTENV_PATH = PROJECT_ROOT / ".env"    # Path to the .env configuration file


# =========================================================================
# Standalone Function Definitions
# =========================================================================
# These are functions that are not methods of any specific class within this module.

# ------------------------
# --- Helper Functions ---
# ------------------------

# --------------------------------------------------------------------------------- _load_env()
def _load_env() -> None:
    """Loads environment variables from the project's .env file.

    Validates that the .env file exists at the expected location and loads
    all environment variables from it into the current process environment.
    This is an internal helper function used by other functions in this module.

    Raises:
        FileNotFoundError: If the .env file is not found at the expected path.

    Examples:
        >>> _load_env()  # Loads variables from PROJECT_ROOT/.env
    """
    if not DOTENV_PATH.exists():
        raise FileNotFoundError(f".env not found at {DOTENV_PATH}")
    load_dotenv(DOTENV_PATH)
# --------------------------------------------------------------------------------- end _load_env()

# --------------------------------------------------------------------------------- _set_env_var()
def _set_env_var(key: str, value: str) -> None:
    """Sets an environment variable in both the .env file and current process.

    Updates the specified environment variable in the .env file and immediately
    sets it in the current process environment. This ensures consistency between
    the persistent configuration and runtime environment.

    Args:
        key (str): The environment variable name to set.
        value (str): The value to assign to the environment variable.

    Examples:
        >>> _set_env_var("APP_ENV", "development")
        >>> _set_env_var("VERBOSE", "true")
    """
    set_key(str(DOTENV_PATH), key, value)
    os.environ[key] = value
# --------------------------------------------------------------------------------- end _set_env_var()

# --------------------------------------------------------------------------------- _get()
def _get(key: str, default: str | None = None) -> str | None:
    """Retrieves an environment variable value with optional default.

    Gets the value of the specified environment variable from the current
    process environment, returning the default value if the variable is not set.

    Args:
        key (str): The environment variable name to retrieve.
        default (str | None, optional): Default value to return if the variable
            is not set. Defaults to None.

    Returns:
        str | None: The environment variable value or the default value.

    Examples:
        >>> app_env = _get("APP_ENV", "development")
        >>> verbose = _get("VERBOSE")
    """
    return os.getenv(key, default)
# --------------------------------------------------------------------------------- end _get()

# --------------------------------------------------------------------------------- _select_credentials()
def _select_credentials(mode: str, force_test_db: bool) -> Tuple[str, str, str]:
    """Selects appropriate Neo4j credentials based on application mode and test database override.

    Determines which set of Neo4j credentials to use based on the application mode
    (development/production) and the force test database flag. The test database
    setting overrides the mode when enabled, providing a way to use test credentials
    regardless of the current application mode.

    Args:
        mode (str): Application mode ("development" or "production").
        force_test_db (bool): If True, use test database credentials regardless of mode.

    Returns:
        Tuple[str, str, str]: A tuple containing (uri, username, password) for Neo4j connection.

    Raises:
        SystemExit: If any required credentials are missing from the .env file.

    Examples:
        >>> uri, user, pwd = _select_credentials("development", False)
        >>> uri, user, pwd = _select_credentials("production", True)  # Uses test DB
    """
    if force_test_db:
        uri = _get("NEO4J_URI_TEST")
        user = _get("NEO4J_USERNAME_TEST")
        pwd = _get("NEO4J_PASSWORD_TEST")
    elif mode == "production":
        uri = _get("NEO4J_URI_PROD")
        user = _get("NEO4J_USERNAME_PROD")
        pwd = _get("NEO4J_PASSWORD_PROD")
    else:
        uri = _get("NEO4J_URI_DEV")
        user = _get("NEO4J_USERNAME_DEV")
        pwd = _get("NEO4J_PASSWORD_DEV")

    if not all([uri, user, pwd]):
        raise SystemExit("Missing Neo4j credentials in .env for the selected mode.")
    return uri, user, pwd
# --------------------------------------------------------------------------------- end _select_credentials()

# ---------------------------------------------
# --- Callable Functions from other modules ---
# ---------------------------------------------

# --------------------------------------------------------------------------------- apply_settings()
def apply_settings(mode: str, force_test_db: bool, verbose: bool | None) -> None:
    """Applies environment configuration settings to .env file and process environment.

    Configures the application environment by setting the application mode,
    database selection, and logging preferences. Updates both the persistent
    .env file and the current process environment to ensure immediate effect.
    Validates input parameters and resolves appropriate database credentials.

    Args:
        mode (str): Application mode ("development" or "production").
        force_test_db (bool): If True, use test database regardless of mode.
        verbose (bool | None): Verbose logging setting. If None, keeps current setting.

    Raises:
        SystemExit: If the mode is invalid or required credentials are missing.

    Examples:
        >>> apply_settings("development", False, True)
        >>> apply_settings("production", False, None)  # Keep current verbose setting
        >>> apply_settings("development", True, False)  # Force test DB
    """
    _load_env()

    # Normalize inputs
    mode = (mode or "development").lower()
    if mode not in {"development", "production"}:
        raise SystemExit("--mode must be 'development' or 'production'")

    # Core flags
    _set_env_var("APP_ENV", mode)
    _set_env_var("FORCE_TEST_DB", str(force_test_db).lower())
    if verbose is not None:
        _set_env_var("VERBOSE", "true" if verbose else "false")

    # Resolve credentials and apply runtime NEO4J_* keys used by services
    uri, user, pwd = _select_credentials(mode, force_test_db)
    _set_env_var("NEO4J_URI", uri)
    _set_env_var("NEO4J_USERNAME", user)
    _set_env_var("NEO4J_PASSWORD", pwd)
# --------------------------------------------------------------------------------- end apply_settings()

# --------------------------------------------------------------------------------- print_status()
def print_status() -> None:
    """Displays the current environment configuration status.

    Loads the current environment settings and displays a formatted summary
    of the application environment, database configuration, and logging settings.
    Provides a clear overview of the current configuration state for verification
    and troubleshooting purposes.

    Examples:
        >>> print_status()
        ============================================================
        APH-IF ENVIRONMENT STATUS (.env)
        ============================================================
        App Environment: DEVELOPMENT
        Force Test DB: false
        Verbose Logging: true
        Neo4j URI: bolt://localhost:7687
        Neo4j Username: neo4j
        ============================================================
    """
    _load_env()
    app_env = _get("APP_ENV", "development")
    force_test_db = _get("FORCE_TEST_DB", "false")
    verbose = _get("VERBOSE", "false")
    uri = _get("NEO4J_URI", "")
    user = _get("NEO4J_USERNAME", "")

    print("\n" + "=" * 60)
    print("APH-IF ENVIRONMENT STATUS (.env)")
    print("=" * 60)
    print(f"App Environment: {app_env.upper()}")
    print(f"Force Test DB: {force_test_db}")
    print(f"Verbose Logging: {verbose}")
    print(f"Neo4j URI: {uri}")
    print(f"Neo4j Username: {user}")
    print("=" * 60)
# --------------------------------------------------------------------------------- end print_status()

# --------------------------
# --- Utility Functions ---
# --------------------------

# --------------------------------------------------------------------------------- parse_args()
def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for environment configuration.

    Defines and parses the command-line interface for the environment setter,
    including options for application mode, test database override, verbose
    logging, and preview mode. Provides comprehensive help text and validation
    for all supported options.

    Returns:
        argparse.Namespace: Parsed command-line arguments with the following attributes:
            - mode: Application mode ("development" or "production")
            - force_test_db: Test database override ("true" or "false")
            - verbose: Verbose logging setting ("true" or "false")
            - preview: Preview mode flag (boolean)

    Examples:
        >>> args = parse_args()
        >>> print(args.mode)  # "development" or "production"
        >>> print(args.preview)  # True or False
    """
    p = argparse.ArgumentParser(description="Set APH-IF environment via .env")
    p.add_argument("--mode", choices=["development", "production"], default=None,
                   help="Target application mode (default: keep current)")
    p.add_argument("--force-test-db", dest="force_test_db", default=None,
                   choices=["true", "false"], help="Override to use test DB")
    p.add_argument("--verbose", default=None, choices=["true", "false"],
                   help="Enable/disable verbose logging")
    p.add_argument("--preview", action="store_true", help="Show current status only")
    return p.parse_args()
# --------------------------------------------------------------------------------- end parse_args()

# --------------------------------------------------------------------------------- main()
def main() -> int:
    """Main entry point for the environment configuration utility.

    Orchestrates the environment configuration process by parsing command-line
    arguments, determining effective settings, and applying the configuration.
    Handles preview mode for status checking and provides comprehensive error
    handling for configuration operations.

    Returns:
        int: Exit code (0 for success, non-zero for failure) suitable for
            command-line usage and automation workflows.

    Examples:
        >>> exit_code = main()  # With --preview flag
        ============================================================
        APH-IF ENVIRONMENT STATUS (.env)
        ============================================================
        App Environment: DEVELOPMENT
        ...
        >>> exit_code = main()  # With configuration changes
        ============================================================
        APH-IF ENVIRONMENT STATUS (.env)
        ============================================================
        App Environment: PRODUCTION
        ...
    """
    args = parse_args()
    if args.preview:
        print_status()
        return 0

    # Determine effective values; if not provided, keep current
    _load_env()
    mode = args.mode or _get("APP_ENV", "development")
    force_test = (_get("FORCE_TEST_DB", "false").lower() == "true")
    if args.force_test_db is not None:
        force_test = (args.force_test_db.lower() == "true")
    verbose = None if args.verbose is None else (args.verbose.lower() == "true")

    apply_settings(mode=mode, force_test_db=force_test, verbose=verbose)
    print_status()
    return 0
# --------------------------------------------------------------------------------- end main()


# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It serves as the entry point for command-line execution of the environment
# configuration utility.

if __name__ == "__main__":  # pragma: no cover
    # --- Direct Execution Entry Point ---
    # Execute the main function and exit with appropriate status code
    # This allows the script to be used in automation workflows and configuration management
    raise SystemExit(main())
# ---------------------------------------------------------------------------------

# =========================================================================
# End of File
# =========================================================================


