# Run Data Processing with Monitoring Module Documentation

## Overview

The `run_data_processing_with_monitoring.py` module provides enhanced monitoring capabilities for the APH-IF data processing pipeline. It wraps the standard pipeline execution with comprehensive API monitoring, real-time progress tracking, error analysis, and performance reporting.

## Purpose

This module enhances the data processing pipeline by providing:

- **Comprehensive API Monitoring**: Tracks all OpenAI and Neo4j API calls with detailed metrics
- **Real-Time Progress Tracking**: Live monitoring dashboard with performance statistics
- **Error Analysis**: Detailed error tracking and categorization for debugging
- **Performance Reporting**: Post-processing analysis with recommendations
- **Resource Monitoring**: System resource usage tracking during processing
- **Data Export**: Monitoring data archival for analysis and optimization

## Architecture

### Core Components

1. **Monitoring Setup**: Configuration of LLM and Neo4j monitors
2. **Pipeline Wrapper**: Subprocess execution with monitoring environment
3. **Real-Time Dashboard**: Optional live monitoring interface
4. **Data Collection**: API call tracking and metrics aggregation
5. **Report Generation**: Post-processing analysis and recommendations
6. **Error Handling**: Comprehensive error tracking and recovery

### Monitoring Flow

```
Setup Monitoring → Configure Environment → Launch Pipeline → Track API Calls → Generate Reports
       ↓                    ↓                   ↓               ↓                ↓
   Monitor Config → Environment Variables → Subprocess → Real-time Data → Analysis & Export
```

## Configuration

### Command Line Arguments

#### Basic Options
- `--monitoring-dir`: Directory for monitoring data (default: `monitoring_data`)
- `--log-level`: Monitoring detail level (`minimal`, `standard`, `detailed`, `debug`)

#### Real-Time Monitoring
- `--watch`: Enable real-time monitoring dashboard
- `--watch-refresh`: Dashboard refresh interval in seconds (default: `5`)

#### Reporting
- `--generate-report`: Generate HTML report after processing
- `--no-cleanup`: Keep temporary monitoring files

#### Pipeline Control
All standard `launch_data_processing.py` arguments are supported:
- `--test-mode`: Force test database usage
- `--initial-only`: Run only initial graph build
- `--augmentation-only`: Run only relationship augmentation
- `--dry-run`: Preview operations without writes

### Environment Variables

#### Monitoring Configuration
- `MONITORING_ENABLED`: Enable monitoring (set automatically)
- `MONITORING_LOG_LEVEL`: Detail level for monitoring
- `MONITORING_DIR`: Directory for monitoring data

#### Standard Pipeline Variables
All standard data processing environment variables are supported.

## Usage

### Basic Monitoring

```powershell
# Run with standard monitoring
cd data_processing
uv run python run_data_processing_with_monitoring.py

# Run with detailed monitoring
uv run python run_data_processing_with_monitoring.py --log-level detailed

# Run with custom monitoring directory
uv run python run_data_processing_with_monitoring.py --monitoring-dir ./my_monitoring
```

### Real-Time Monitoring

```powershell
# Enable real-time dashboard
uv run python run_data_processing_with_monitoring.py --watch

# Custom refresh interval
uv run python run_data_processing_with_monitoring.py --watch --watch-refresh 10

# Real-time monitoring with report generation
uv run python run_data_processing_with_monitoring.py --watch --generate-report
```

### Test Mode with Monitoring

```powershell
# Test mode with monitoring
uv run python run_data_processing_with_monitoring.py --test-mode --watch

# Test mode with detailed logging
uv run python run_data_processing_with_monitoring.py --test-mode --log-level debug
```

### Pipeline-Specific Monitoring

```powershell
# Monitor only initial graph build
uv run python run_data_processing_with_monitoring.py --initial-only --watch

# Monitor only relationship augmentation
uv run python run_data_processing_with_monitoring.py --augmentation-only --generate-report

# Dry run with monitoring
uv run python run_data_processing_with_monitoring.py --dry-run --log-level detailed
```

## API Reference

### Core Functions

#### setup_monitoring(monitoring_dir: Path, log_level: LogLevel) -> tuple

```python
def setup_monitoring(monitoring_dir: Path, log_level: LogLevel) -> tuple:
    """Setup monitoring configuration.
    
    Args:
        monitoring_dir: Directory for monitoring data
        log_level: Detail level for monitoring
        
    Returns:
        tuple: (llm_monitor, neo4j_monitor) instances
    """
```

#### run_data_processing_pipeline(args) -> int

```python
def run_data_processing_pipeline(args) -> int:
    """Run the data processing pipeline with monitoring.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        int: Exit code (0 for success, non-zero for failure)
    """
```

### Monitoring Features

#### API Call Tracking

```python
# Automatic tracking of:
# - OpenAI API calls (embeddings, chat completions)
# - Neo4j operations (queries, transactions)
# - Request/response timing
# - Token usage and costs
# - Error rates and types
```

#### Real-Time Statistics

```python
# Live dashboard shows:
# - Total API calls
# - Success/failure rates
# - Average response times
# - Token usage trends
# - Error categorization
```

#### Performance Metrics

```python
# Collected metrics include:
# - API call duration
# - Token consumption
# - Database query performance
# - Error frequency and types
# - Resource utilization
```

## Monitoring Data Structure

### LLM API Monitoring

```json
{
  "timestamp": "2025-01-18T10:30:45.123Z",
  "operation": "chat_completion",
  "model": "gpt-5-nano",
  "duration_ms": 1250,
  "tokens_used": 150,
  "success": true,
  "error": null,
  "request_size": 1024,
  "response_size": 512
}
```

### Neo4j Operation Monitoring

```json
{
  "timestamp": "2025-01-18T10:30:45.456Z",
  "operation": "MERGE",
  "query": "MERGE (d:Document {doc_id: $doc_id})",
  "duration_ms": 45,
  "records_returned": 1,
  "success": true,
  "error": null
}
```

### Monitoring Files

```
monitoring_data/
├── llm_api_calls.jsonl          # LLM API call logs
├── neo4j_operations.jsonl       # Neo4j operation logs
├── pipeline_report_*.html       # Generated reports
└── monitoring_summary.json      # Aggregated statistics
```

## Real-Time Dashboard

### Dashboard Features

- **Live Statistics**: Real-time API call counts and performance metrics
- **Error Tracking**: Immediate notification of errors and failures
- **Progress Monitoring**: Visual progress indicators for pipeline stages
- **Resource Usage**: System resource monitoring during processing
- **Cost Tracking**: Estimated API costs and token usage

### Dashboard Output Example

```
APH-IF Data Processing - Real-Time Monitor
==========================================
Pipeline Status: RUNNING
Elapsed Time: 00:05:23

API Statistics:
   Total Calls: 1,247
   LLM Calls: 856 (68.6%)
   Neo4j Operations: 391 (31.4%)
   Success Rate: 98.7%
   Avg Duration: 245ms

Current Activity:
   Processing: entity_extraction
   Last API Call: 2s ago
   Errors: 3 (0.2%)

Resource Usage:
   CPU: 45%
   Memory: 2.1GB
   Network: 125KB/s
```

## Report Generation

### HTML Report Features

- **Executive Summary**: High-level pipeline performance overview
- **API Performance**: Detailed analysis of API call patterns
- **Error Analysis**: Categorized error reports with recommendations
- **Cost Analysis**: Token usage and estimated costs
- **Performance Trends**: Timeline analysis of processing performance
- **Recommendations**: Optimization suggestions based on monitoring data

### Report Sections

1. **Pipeline Overview**: Duration, success rate, major milestones
2. **API Performance**: Call volumes, response times, error rates
3. **Resource Utilization**: CPU, memory, and network usage patterns
4. **Error Analysis**: Error categorization and troubleshooting guidance
5. **Cost Analysis**: Token usage breakdown and cost estimates
6. **Optimization Recommendations**: Performance improvement suggestions

## Error Handling and Recovery

### Error Categories

1. **API Errors**: OpenAI rate limits, authentication failures
2. **Database Errors**: Neo4j connection issues, query failures
3. **System Errors**: Resource exhaustion, network connectivity
4. **Pipeline Errors**: Processing failures, data corruption

### Recovery Mechanisms

```python
# Automatic error handling:
# - API retry logic with exponential backoff
# - Database connection recovery
# - Graceful degradation for non-critical errors
# - Comprehensive error logging for debugging
```

### Monitoring During Errors

```python
# Error monitoring includes:
# - Error categorization and frequency
# - Context capture for debugging
# - Performance impact analysis
# - Recovery time tracking
```

## Performance Optimization

### Monitoring-Based Optimization

```python
# Use monitoring data to optimize:
# - API call batching strategies
# - Database query performance
# - Resource allocation
# - Error prevention
```

### Performance Metrics

- **API Efficiency**: Calls per minute, token efficiency
- **Database Performance**: Query response times, connection usage
- **Resource Utilization**: CPU, memory, and network efficiency
- **Error Rates**: Success rates and error frequency

## Integration Points

### With Core Pipeline

```python
# Seamless integration with:
# - launch_data_processing.py
# - All processing modules
# - Environment management
# - Configuration systems
```

### With Monitoring Infrastructure

```python
# Integrates with:
# - common.api_monitor
# - common.monitoring_dashboard
# - External monitoring systems
# - Log aggregation platforms
```

## Best Practices

### Development Workflow

1. **Start with Monitoring**: Always use monitoring during development
2. **Use Real-Time Dashboard**: Enable `--watch` for immediate feedback
3. **Analyze Reports**: Review generated reports for optimization opportunities
4. **Monitor Resource Usage**: Watch for memory leaks and performance issues
5. **Track API Costs**: Monitor token usage and associated costs

### Production Deployment

1. **Enable Monitoring**: Always run with monitoring in production
2. **Archive Monitoring Data**: Keep monitoring data for trend analysis
3. **Set Up Alerts**: Configure alerts for error rates and performance issues
4. **Regular Analysis**: Review monitoring reports for optimization
5. **Cost Management**: Track and optimize API usage costs

### Troubleshooting

1. **Check Monitoring Logs**: Review detailed API call logs for errors
2. **Analyze Error Patterns**: Look for recurring error types
3. **Monitor Resource Usage**: Check for resource exhaustion issues
4. **Review Performance Trends**: Identify performance degradation patterns
5. **Use Debug Logging**: Enable detailed logging for complex issues

## Examples

### Basic Monitoring

```powershell
# Standard monitoring with report
uv run python run_data_processing_with_monitoring.py --generate-report

# Detailed monitoring for debugging
uv run python run_data_processing_with_monitoring.py --log-level debug --watch
```

### Test Environment Monitoring

```powershell
# Test mode with comprehensive monitoring
uv run python run_data_processing_with_monitoring.py --test-mode --watch --generate-report

# Quick test with minimal monitoring
uv run python run_data_processing_with_monitoring.py --test-mode --log-level minimal
```

### Production Monitoring

```powershell
# Production run with full monitoring
uv run python run_data_processing_with_monitoring.py --log-level standard --generate-report --monitoring-dir ./prod_monitoring

# Long-running process with real-time monitoring
uv run python run_data_processing_with_monitoring.py --watch --watch-refresh 30
```

### Step-Specific Monitoring

```powershell
# Monitor only initial build
uv run python run_data_processing_with_monitoring.py --initial-only --log-level detailed --watch

# Monitor relationship augmentation with cost tracking
uv run python run_data_processing_with_monitoring.py --augmentation-only --generate-report
```

## Output Example

```
APH-IF Data Processing with Enhanced Monitoring
================================================================================

Monitoring configured:
   Directory: monitoring_data
   Log Level: standard
   LLM Monitor: monitoring_data/llm_api_calls.jsonl
   Neo4j Monitor: monitoring_data/neo4j_operations.jsonl

Starting real-time monitoring dashboard...
   Dashboard PID: 12345

Starting Data Processing Pipeline...
================================================================================
[Pipeline execution with real-time monitoring...]

================================================================================
Pipeline Execution Completed
   Duration: 0:08:45
   Exit Code: 0
================================================================================

Generating Monitoring Report...
   Report generated: monitoring_data/pipeline_report_20250118_103045.html

Final Statistics:
   Total API Calls: 2,156
   LLM Calls: 1,423
   Neo4j Operations: 733
   Success Rate: 99.2%
   Avg Duration: 187.3ms

Key Recommendations:
   1. Consider batching smaller API requests for better efficiency
   2. Database query performance is optimal
   3. No significant errors detected
```

This module provides essential monitoring capabilities for production APH-IF deployments, enabling comprehensive tracking, analysis, and optimization of the data processing pipeline.
