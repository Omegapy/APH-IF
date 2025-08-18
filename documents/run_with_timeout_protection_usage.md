# Run with Timeout Protection Module Documentation

## Overview

The `run_with_timeout_protection.py` module provides comprehensive timeout protection and process management for the APH-IF data processing pipeline. It implements multiple layers of protection against hanging processes, connection pool exhaustion, and system resource issues that can occur during long-running data processing operations.

## Purpose

This module addresses common issues in data processing by providing:

- **Process Timeout Monitoring**: Automatic termination of processes that exceed time limits
- **Hang Detection**: Detection and recovery from processes that stop producing output
- **Connection Pool Management**: Prevention of connection pool exhaustion
- **Automatic Process Cleanup**: Proper termination of parent and child processes
- **Environment Stabilization**: Configuration optimizations for reliable processing
- **Enhanced Error Handling**: Comprehensive error recovery and reporting

## Architecture

### Core Components

1. **TimeoutProtection Class**: Main process monitoring and protection logic
2. **Process Monitoring**: Background thread monitoring for timeouts and hangs
3. **Environment Setup**: Stability-focused environment configuration
4. **Process Management**: Safe process creation, monitoring, and termination
5. **Output Tracking**: Real-time output monitoring for hang detection
6. **Cleanup Mechanisms**: Comprehensive process tree termination

### Protection Layers

```
Command Execution → Process Creation → Timeout Monitoring → Hang Detection → Process Cleanup
       ↓                  ↓                 ↓                  ↓               ↓
   Environment Setup → Subprocess → Background Monitor → Output Tracking → Safe Termination
```

## Configuration

### Command Line Arguments

#### Timeout Control
- `--timeout`: Timeout in minutes (default: `60`)

#### Processing Control
- `--test-mode`: Run in test mode with safe defaults
- `--initial-only`: Run only initial graph build step

#### Monitoring Integration
- `--monitoring-dir`: Directory for monitoring data (default: `monitoring_logs`)
- `--log-level`: Monitoring log level (default: `detailed`)

### Environment Variables

#### Stability Configuration (Set Automatically)
- `HTTPX_TIMEOUT`: HTTP request timeout (120 seconds)
- `OPENAI_TIMEOUT`: OpenAI API timeout (120 seconds)
- `HTTPX_POOL_CONNECTIONS`: Connection pool size (10 connections)
- `HTTPX_POOL_MAXSIZE`: Maximum pool size (10 connections)
- `LANGCHAIN_ASYNC`: Disable async mode for stability (`false`)

## Usage

### Basic Usage

```powershell
# Run with default 60-minute timeout
cd data_processing
uv run python run_with_timeout_protection.py

# Run with custom timeout
uv run python run_with_timeout_protection.py --timeout 90

# Run in test mode
uv run python run_with_timeout_protection.py --test-mode --timeout 30
```

### Specific Processing Steps

```powershell
# Run only initial graph build with protection
uv run python run_with_timeout_protection.py --initial-only --timeout 45

# Run with custom monitoring settings
uv run python run_with_timeout_protection.py --monitoring-dir ./custom_logs --log-level debug
```

### Production Usage

```powershell
# Production run with extended timeout
uv run python run_with_timeout_protection.py --timeout 120 --log-level standard

# Long-running process with comprehensive monitoring
uv run python run_with_timeout_protection.py --timeout 180 --monitoring-dir ./prod_monitoring
```

## API Reference

### TimeoutProtection Class

#### Constructor

```python
def __init__(self, timeout_minutes=30):
    """Initialize timeout protection.
    
    Args:
        timeout_minutes: Maximum process runtime in minutes
    """
```

#### Core Methods

##### run_with_protection(cmd_args, cwd=None) -> int

```python
def run_with_protection(self, cmd_args, cwd=None) -> int:
    """Run command with timeout protection.
    
    Args:
        cmd_args: Command arguments list
        cwd: Working directory (optional)
        
    Returns:
        int: Process exit code
    """
```

**Features:**
- Real-time output monitoring
- Background timeout monitoring
- Automatic process cleanup
- Comprehensive error handling

### Protection Mechanisms

#### Total Timeout Protection

```python
# Monitors total process runtime
total_duration = now - self.start_time
if total_duration > timedelta(minutes=self.timeout_minutes):
    print(f"TIMEOUT: Process exceeded {self.timeout_minutes} minutes")
    self._kill_process()
```

#### Output Timeout Protection

```python
# Monitors output activity
output_duration = now - self.last_output_time
if output_duration > timedelta(minutes=15):
    print(f"HANG DETECTED: No output for {output_duration}")
    self._kill_process()
```

#### Process Tree Cleanup

```python
def _kill_process(self):
    """Kill the process and all children."""
    # Uses psutil to terminate entire process tree
    # Graceful termination followed by force kill if needed
```

### Environment Stabilization

#### setup_environment_for_stability()

```python
def setup_environment_for_stability():
    """Set environment variables to improve stability."""
    # HTTP connection settings
    os.environ['HTTPX_TIMEOUT'] = '120'
    os.environ['OPENAI_TIMEOUT'] = '120'
    
    # Reduce connection pool size to prevent exhaustion
    os.environ['HTTPX_POOL_CONNECTIONS'] = '10'
    os.environ['HTTPX_POOL_MAXSIZE'] = '10'
    
    # Force synchronous mode for stability
    os.environ['LANGCHAIN_ASYNC'] = 'false'
```

## Timeout Mechanisms

### Total Process Timeout

- **Default**: 60 minutes (configurable)
- **Purpose**: Prevent runaway processes
- **Action**: Automatic process termination
- **Monitoring**: Checked every 30 seconds

### Output Activity Timeout

- **Warning Threshold**: 10 minutes without output
- **Termination Threshold**: 15 minutes without output
- **Purpose**: Detect hung processes that appear active but aren't progressing
- **Action**: Process termination and cleanup

### Connection Timeout

- **HTTP Requests**: 120 seconds
- **OpenAI API**: 120 seconds
- **Purpose**: Prevent hanging on network operations
- **Fallback**: Automatic retry and error handling

## Process Management

### Process Creation

```python
# Line-buffered subprocess with real-time output
self.process = subprocess.Popen(
    cmd_args,
    cwd=cwd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,  # Line buffered
    universal_newlines=True
)
```

### Output Monitoring

```python
# Real-time output processing with activity tracking
while True:
    output = self.process.stdout.readline()
    if output == '' and self.process.poll() is not None:
        break
    if output:
        print(output.strip())
        self.last_output_time = datetime.now()  # Track activity
```

### Process Cleanup

```python
# Comprehensive process tree termination
# 1. Terminate all child processes
# 2. Terminate parent process
# 3. Wait for graceful shutdown
# 4. Force kill if necessary
```

## Safety Features

### Graceful Termination

```python
# Multi-stage termination process:
# 1. SIGTERM to all processes
# 2. Wait for graceful shutdown (5 seconds)
# 3. SIGKILL for any remaining processes
```

### Error Recovery

```python
# Comprehensive error handling:
# - Keyboard interrupt handling
# - Process creation errors
# - Monitoring thread errors
# - Cleanup failures
```

### Resource Protection

```python
# Environment optimizations:
# - Limited connection pools
# - Synchronous processing mode
# - Extended timeouts for stability
# - Memory-efficient buffering
```

## Integration Points

### With Monitoring System

```python
# Integrates with run_data_processing_with_monitoring.py
cmd_args = [
    sys.executable, 
    'run_data_processing_with_monitoring.py',
    '--monitoring-dir', args.monitoring_dir,
    '--log-level', args.log_level
]
```

### With Process Health Monitoring

```python
# Works with monitor_process_health.py for comprehensive monitoring
# Provides process-level protection while health monitor provides system-level monitoring
```

### With Stuck Process Cleanup

```python
# Integrates with kill_stuck_process.py for manual cleanup
if return_code != 0:
    print("To kill any remaining stuck processes, run:")
    print("   uv run python kill_stuck_process.py")
```

## Error Handling

### Timeout Errors

```python
# Total timeout exceeded
if total_duration > timedelta(minutes=self.timeout_minutes):
    print(f"TIMEOUT: Process exceeded {self.timeout_minutes} minutes")
    self._kill_process()
    return 1
```

### Hang Detection

```python
# Output timeout (hang detection)
if output_duration > timedelta(minutes=15):
    print(f"HANG DETECTED: No output for {output_duration}")
    self._kill_process()
    return 1
```

### Process Errors

```python
# Process creation and execution errors
except Exception as e:
    print(f"Process error: {e}")
    self._kill_process()
    return 1
```

### Cleanup Errors

```python
# Graceful error handling during cleanup
except Exception as e:
    print(f"Error killing process: {e}")
    # Continue with fallback cleanup methods
```

## Best Practices

### Development Usage

1. **Start with Test Mode**: Use `--test-mode` for initial testing
2. **Use Shorter Timeouts**: Set appropriate timeouts for development data
3. **Monitor Output**: Watch for hang warnings and adjust processing
4. **Check Logs**: Review monitoring logs for performance issues

### Production Usage

1. **Set Appropriate Timeouts**: Allow sufficient time for large datasets
2. **Monitor System Resources**: Ensure adequate memory and CPU
3. **Use Monitoring**: Enable comprehensive monitoring for production runs
4. **Plan for Recovery**: Have procedures for handling timeout situations

### Troubleshooting

1. **Check System Resources**: Ensure adequate memory and CPU availability
2. **Review Network Connectivity**: Verify stable internet connection
3. **Monitor API Limits**: Check for rate limiting or quota issues
4. **Analyze Logs**: Review monitoring logs for error patterns

## Examples

### Basic Protection

```powershell
# Standard timeout protection
uv run python run_with_timeout_protection.py

# Custom timeout for large datasets
uv run python run_with_timeout_protection.py --timeout 120
```

### Test Environment

```powershell
# Test mode with short timeout
uv run python run_with_timeout_protection.py --test-mode --timeout 15

# Test only initial build
uv run python run_with_timeout_protection.py --test-mode --initial-only --timeout 30
```

### Production Environment

```powershell
# Production run with extended timeout
uv run python run_with_timeout_protection.py --timeout 180 --log-level standard

# Long-running process with comprehensive monitoring
uv run python run_with_timeout_protection.py --timeout 240 --monitoring-dir ./prod_logs
```

### Recovery Scenarios

```powershell
# If process times out or hangs, clean up manually
uv run python kill_stuck_process.py

# Check system health
uv run python monitor_process_health.py
```

## Output Example

```
Data Processing with Timeout Protection
============================================================

Environment configured for stability:
   - HTTP timeout: 120s
   - Connection pool: 10 connections
   - Async mode: disabled

Starting process with 60-minute timeout protection
Command: python run_data_processing_with_monitoring.py --monitoring-dir monitoring_logs --log-level detailed
Working directory: P:\Projects\APH-IF\APH-IF-Dev\data_processing

[Real-time process output...]

⚠️  WARNING: No output for 0:11:23
Process may be hanging...

Process completed in 0:45:32
Exit code: 0
```

## Common Issues and Solutions

### Process Hangs

**Symptoms**: No output for extended periods
**Solution**: Automatic termination after 15 minutes of inactivity

### Connection Pool Exhaustion

**Symptoms**: HTTP connection errors
**Solution**: Limited connection pool size (10 connections)

### Memory Issues

**Symptoms**: Process slowdown or crashes
**Solution**: Monitor system resources and adjust processing parameters

### API Rate Limits

**Symptoms**: API timeout errors
**Solution**: Extended timeouts (120 seconds) and retry logic

This module provides essential protection for production APH-IF deployments, ensuring reliable processing even with large datasets and potential network or system issues.
