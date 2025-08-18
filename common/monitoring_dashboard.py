"""
API Monitoring Dashboard and Reporting

This module provides utilities to analyze, visualize, and report on API monitoring data
collected during APH-IF data processing operations.

Features:
- Load and analyze monitoring data from JSON log files
- Generate comprehensive reports on API usage and performance
- Identify bottlenecks, errors, and optimization opportunities
- Export reports in multiple formats (JSON, CSV, HTML)
- Real-time monitoring statistics

Usage:
    from common.monitoring_dashboard import MonitoringDashboard
    
    # Create dashboard
    dashboard = MonitoringDashboard()
    
    # Load monitoring data
    dashboard.load_data("monitoring_logs/")
    
    # Generate reports
    report = dashboard.generate_comprehensive_report()
    dashboard.export_report(report, "monitoring_report.html", format="html")
    
    # Get real-time stats
    stats = dashboard.get_realtime_stats()
"""

import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import statistics
from collections import defaultdict, Counter

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


@dataclass
class MonitoringReport:
    """Comprehensive monitoring report data structure."""
    generation_time: datetime
    time_range: Dict[str, datetime]
    summary: Dict[str, Any]
    api_breakdown: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    error_analysis: Dict[str, Any]
    recommendations: List[str]
    raw_data: Optional[Dict[str, Any]] = None


class MonitoringDashboard:
    """Dashboard for analyzing and reporting on API monitoring data."""
    
    def __init__(self):
        self.data: List[Dict[str, Any]] = []
        self.llm_data: List[Dict[str, Any]] = []
        self.neo4j_data: List[Dict[str, Any]] = []
        self.loaded_files: List[Path] = []
    
    def load_data(self, source: Union[Path, str, List[Path]]) -> int:
        """
        Load monitoring data from files or directory.
        
        Args:
            source: Path to file, directory, or list of files
            
        Returns:
            Number of records loaded
        """
        if isinstance(source, (str, Path)):
            source = Path(source)
            if source.is_dir():
                files = list(source.glob("*.jsonl")) + list(source.glob("*.json"))
            else:
                files = [source]
        else:
            files = [Path(f) for f in source]
        
        records_loaded = 0
        
        for file_path in files:
            if not file_path.exists():
                continue
                
            try:
                if file_path.suffix == '.jsonl':
                    # JSON Lines format
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                self._categorize_record(record)
                                records_loaded += 1
                else:
                    # Regular JSON format
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            for record in data:
                                self._categorize_record(record)
                                records_loaded += 1
                        else:
                            self._categorize_record(data)
                            records_loaded += 1
                
                self.loaded_files.append(file_path)
                
            except Exception as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        return records_loaded
    
    def _categorize_record(self, record: Dict[str, Any]):
        """Categorize a monitoring record by API type."""
        self.data.append(record)
        
        api_type = record.get('api_type', '').lower()
        if api_type == 'llm':
            self.llm_data.append(record)
        elif api_type == 'neo4j':
            self.neo4j_data.append(record)
    
    def get_realtime_stats(self) -> Dict[str, Any]:
        """Get real-time monitoring statistics."""
        if not self.data:
            return {"error": "No data loaded"}
        
        now = datetime.now()
        recent_data = [
            r for r in self.data 
            if self._parse_datetime(r.get('start_time')) and 
               (now - self._parse_datetime(r.get('start_time'))).total_seconds() < 3600
        ]
        
        return {
            "total_records": len(self.data),
            "recent_hour_records": len(recent_data),
            "llm_calls": len(self.llm_data),
            "neo4j_operations": len(self.neo4j_data),
            "success_rate": self._calculate_success_rate(self.data),
            "avg_duration_ms": self._calculate_avg_duration(self.data),
            "last_updated": now.isoformat()
        }
    
    def generate_comprehensive_report(self, include_raw_data: bool = False) -> MonitoringReport:
        """Generate a comprehensive monitoring report."""
        if not self.data:
            raise ValueError("No monitoring data loaded")
        
        # Time range analysis
        start_times = [self._parse_datetime(r.get('start_time')) for r in self.data]
        start_times = [t for t in start_times if t is not None]
        
        time_range = {
            "start": min(start_times) if start_times else datetime.now(),
            "end": max(start_times) if start_times else datetime.now()
        }
        
        # Summary statistics
        summary = {
            "total_api_calls": len(self.data),
            "llm_calls": len(self.llm_data),
            "neo4j_operations": len(self.neo4j_data),
            "success_rate": self._calculate_success_rate(self.data),
            "total_duration_hours": (time_range["end"] - time_range["start"]).total_seconds() / 3600,
            "calls_per_hour": len(self.data) / max(1, (time_range["end"] - time_range["start"]).total_seconds() / 3600)
        }
        
        # API breakdown
        api_breakdown = {
            "llm": self._analyze_llm_data(),
            "neo4j": self._analyze_neo4j_data()
        }
        
        # Performance metrics
        performance_metrics = self._analyze_performance()
        
        # Error analysis
        error_analysis = self._analyze_errors()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(summary, performance_metrics, error_analysis)
        
        return MonitoringReport(
            generation_time=datetime.now(),
            time_range=time_range,
            summary=summary,
            api_breakdown=api_breakdown,
            performance_metrics=performance_metrics,
            error_analysis=error_analysis,
            recommendations=recommendations,
            raw_data={"all_data": self.data} if include_raw_data else None
        )
    
    def _analyze_llm_data(self) -> Dict[str, Any]:
        """Analyze LLM API call data."""
        if not self.llm_data:
            return {"total_calls": 0}
        
        models = [r.get('model') for r in self.llm_data if r.get('model')]
        operations = [r.get('operation') for r in self.llm_data if r.get('operation')]
        tokens = [r.get('total_tokens') for r in self.llm_data if r.get('total_tokens')]
        durations = [r.get('duration_ms') for r in self.llm_data if r.get('duration_ms')]
        
        return {
            "total_calls": len(self.llm_data),
            "success_rate": self._calculate_success_rate(self.llm_data),
            "models_used": dict(Counter(models)),
            "operations": dict(Counter(operations)),
            "total_tokens": sum(tokens),
            "avg_tokens_per_call": statistics.mean(tokens) if tokens else 0,
            "avg_duration_ms": statistics.mean(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "estimated_cost_usd": self._estimate_llm_cost(self.llm_data)
        }
    
    def _analyze_neo4j_data(self) -> Dict[str, Any]:
        """Analyze Neo4j operation data."""
        if not self.neo4j_data:
            return {"total_operations": 0}
        
        operations = [r.get('operation') for r in self.neo4j_data if r.get('operation')]
        records_returned = [r.get('records_returned') for r in self.neo4j_data if r.get('records_returned')]
        records_affected = [r.get('records_affected') for r in self.neo4j_data if r.get('records_affected')]
        durations = [r.get('duration_ms') for r in self.neo4j_data if r.get('duration_ms')]
        
        return {
            "total_operations": len(self.neo4j_data),
            "success_rate": self._calculate_success_rate(self.neo4j_data),
            "operation_types": dict(Counter(operations)),
            "total_records_returned": sum(records_returned),
            "total_records_affected": sum(records_affected),
            "avg_records_per_query": statistics.mean(records_returned) if records_returned else 0,
            "avg_duration_ms": statistics.mean(durations) if durations else 0,
            "max_duration_ms": max(durations) if durations else 0,
            "slow_queries": len([d for d in durations if d and d > 1000])  # > 1 second
        }
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall performance metrics."""
        durations = [r.get('duration_ms') for r in self.data if r.get('duration_ms')]
        
        if not durations:
            return {"no_duration_data": True}
        
        return {
            "avg_duration_ms": statistics.mean(durations),
            "median_duration_ms": statistics.median(durations),
            "p95_duration_ms": self._percentile(durations, 95),
            "p99_duration_ms": self._percentile(durations, 99),
            "min_duration_ms": min(durations),
            "max_duration_ms": max(durations),
            "slow_calls": len([d for d in durations if d > 5000]),  # > 5 seconds
            "very_slow_calls": len([d for d in durations if d > 30000])  # > 30 seconds
        }
    
    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze error patterns."""
        errors = [r for r in self.data if not r.get('success', True)]
        
        if not errors:
            return {"total_errors": 0, "error_rate": 0.0}
        
        error_types = [r.get('error_type') for r in errors if r.get('error_type')]
        error_messages = [r.get('error_message') for r in errors if r.get('error_message')]
        
        return {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(self.data),
            "error_types": dict(Counter(error_types)),
            "common_error_messages": dict(Counter(error_messages)[:10]),  # Top 10
            "errors_by_api": {
                "llm": len([r for r in errors if r.get('api_type') == 'llm']),
                "neo4j": len([r for r in errors if r.get('api_type') == 'neo4j'])
            }
        }
    
    def _generate_recommendations(self, summary: Dict, performance: Dict, errors: Dict) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Performance recommendations
        if performance.get('avg_duration_ms', 0) > 2000:
            recommendations.append("Average API response time is high (>2s). Consider optimizing queries or using caching.")
        
        if performance.get('slow_calls', 0) > summary.get('total_api_calls', 0) * 0.1:
            recommendations.append("More than 10% of calls are slow (>5s). Review query complexity and database indexes.")
        
        # Error rate recommendations
        error_rate = errors.get('error_rate', 0)
        if error_rate > 0.05:
            recommendations.append(f"Error rate is {error_rate:.1%}. Implement retry logic and better error handling.")
        
        # LLM-specific recommendations
        llm_calls = summary.get('llm_calls', 0)
        if llm_calls > 1000:
            recommendations.append("High LLM usage detected. Consider batching requests or using smaller models for simple tasks.")
        
        # Neo4j-specific recommendations
        neo4j_ops = summary.get('neo4j_operations', 0)
        if neo4j_ops > 10000:
            recommendations.append("High Neo4j operation count. Consider using batch operations and connection pooling.")
        
        if not recommendations:
            recommendations.append("System performance looks good! No immediate optimizations needed.")
        
        return recommendations
    
    def _calculate_success_rate(self, data: List[Dict]) -> float:
        """Calculate success rate for a dataset."""
        if not data:
            return 0.0
        successful = len([r for r in data if r.get('success', True)])
        return successful / len(data)
    
    def _calculate_avg_duration(self, data: List[Dict]) -> float:
        """Calculate average duration for a dataset."""
        durations = [r.get('duration_ms') for r in data if r.get('duration_ms')]
        return statistics.mean(durations) if durations else 0.0
    
    def _parse_datetime(self, dt_str: Optional[str]) -> Optional[datetime]:
        """Parse datetime string safely."""
        if not dt_str:
            return None
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except:
            return None
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _estimate_llm_cost(self, llm_data: List[Dict]) -> float:
        """Estimate LLM usage cost (approximate)."""
        total_cost = 0.0
        
        # Simplified pricing (actual prices may vary)
        pricing = {
            "gpt-4": 0.03,  # per 1K tokens
            "gpt-4-turbo": 0.01,
            "gpt-3.5-turbo": 0.002,
            "text-embedding-3-large": 0.00013,
            "text-embedding-3-small": 0.00002,
        }
        
        for record in llm_data:
            model = record.get('model', '')
            tokens = record.get('total_tokens', 0)
            
            # Find matching pricing
            price_per_1k = 0.01  # default
            for model_name, price in pricing.items():
                if model_name in model.lower():
                    price_per_1k = price
                    break
            
            total_cost += (tokens / 1000) * price_per_1k
        
        return total_cost

    def export_report(self, report: MonitoringReport, output_path: Union[Path, str],
                     format: str = "json") -> bool:
        """
        Export monitoring report to file.

        Args:
            report: MonitoringReport to export
            output_path: Output file path
            format: Export format ("json", "html", "csv")

        Returns:
            True if successful, False otherwise
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if format.lower() == "json":
                self._export_json(report, output_path)
            elif format.lower() == "html":
                self._export_html(report, output_path)
            elif format.lower() == "csv":
                self._export_csv(report, output_path)
            else:
                raise ValueError(f"Unsupported export format: {format}")

            return True

        except Exception as e:
            print(f"Error exporting report: {e}")
            return False

    def _export_json(self, report: MonitoringReport, output_path: Path):
        """Export report as JSON."""
        data = {
            "generation_time": report.generation_time.isoformat(),
            "time_range": {
                "start": report.time_range["start"].isoformat(),
                "end": report.time_range["end"].isoformat()
            },
            "summary": report.summary,
            "api_breakdown": report.api_breakdown,
            "performance_metrics": report.performance_metrics,
            "error_analysis": report.error_analysis,
            "recommendations": report.recommendations
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def _export_html(self, report: MonitoringReport, output_path: Path):
        """Export report as HTML."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>APH-IF API Monitoring Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
        .error {{ color: #d32f2f; }}
        .success {{ color: #388e3c; }}
        .warning {{ color: #f57c00; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .recommendations {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>APH-IF API Monitoring Report</h1>
        <p>Generated: {report.generation_time.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Time Range: {report.time_range['start'].strftime('%Y-%m-%d %H:%M')} to {report.time_range['end'].strftime('%Y-%m-%d %H:%M')}</p>
    </div>

    <div class="section">
        <h2>Summary</h2>
        <div class="metric">Total API Calls: <strong>{report.summary['total_api_calls']:,}</strong></div>
        <div class="metric">LLM Calls: <strong>{report.summary['llm_calls']:,}</strong></div>
        <div class="metric">Neo4j Operations: <strong>{report.summary['neo4j_operations']:,}</strong></div>
        <div class="metric">Success Rate: <strong class="{'success' if report.summary['success_rate'] > 0.95 else 'warning' if report.summary['success_rate'] > 0.9 else 'error'}">{report.summary['success_rate']:.1%}</strong></div>
        <div class="metric">Calls/Hour: <strong>{report.summary['calls_per_hour']:.1f}</strong></div>
    </div>

    <div class="section">
        <h2>LLM API Analysis</h2>
        <div class="metric">Total Calls: {report.api_breakdown['llm']['total_calls']:,}</div>
        <div class="metric">Success Rate: {report.api_breakdown['llm']['success_rate']:.1%}</div>
        <div class="metric">Total Tokens: {report.api_breakdown['llm']['total_tokens']:,}</div>
        <div class="metric">Avg Duration: {report.api_breakdown['llm']['avg_duration_ms']:.1f}ms</div>
        <div class="metric">Estimated Cost: ${report.api_breakdown['llm']['estimated_cost_usd']:.2f}</div>
    </div>

    <div class="section">
        <h2>Neo4j Analysis</h2>
        <div class="metric">Total Operations: {report.api_breakdown['neo4j']['total_operations']:,}</div>
        <div class="metric">Success Rate: {report.api_breakdown['neo4j']['success_rate']:.1%}</div>
        <div class="metric">Records Returned: {report.api_breakdown['neo4j']['total_records_returned']:,}</div>
        <div class="metric">Records Affected: {report.api_breakdown['neo4j']['total_records_affected']:,}</div>
        <div class="metric">Avg Duration: {report.api_breakdown['neo4j']['avg_duration_ms']:.1f}ms</div>
        <div class="metric">Slow Queries: {report.api_breakdown['neo4j']['slow_queries']}</div>
    </div>

    <div class="section">
        <h2>Performance Metrics</h2>
        <div class="metric">Avg Duration: {report.performance_metrics.get('avg_duration_ms', 0):.1f}ms</div>
        <div class="metric">P95 Duration: {report.performance_metrics.get('p95_duration_ms', 0):.1f}ms</div>
        <div class="metric">P99 Duration: {report.performance_metrics.get('p99_duration_ms', 0):.1f}ms</div>
        <div class="metric">Slow Calls: {report.performance_metrics.get('slow_calls', 0)}</div>
    </div>

    <div class="section">
        <h2>Error Analysis</h2>
        <div class="metric">Total Errors: <span class="{'error' if report.error_analysis['total_errors'] > 0 else 'success'}">{report.error_analysis['total_errors']}</span></div>
        <div class="metric">Error Rate: <span class="{'error' if report.error_analysis['error_rate'] > 0.05 else 'warning' if report.error_analysis['error_rate'] > 0.01 else 'success'}">{report.error_analysis['error_rate']:.1%}</span></div>
    </div>

    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
            {''.join(f'<li>{rec}</li>' for rec in report.recommendations)}
        </ul>
    </div>
</body>
</html>
        """

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _export_csv(self, report: MonitoringReport, output_path: Path):
        """Export report summary as CSV."""
        if not PANDAS_AVAILABLE:
            # Fallback to manual CSV writing
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])

                # Summary metrics
                for key, value in report.summary.items():
                    writer.writerow([key, value])

                # Performance metrics
                for key, value in report.performance_metrics.items():
                    writer.writerow([f'performance_{key}', value])

                # Error metrics
                for key, value in report.error_analysis.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            writer.writerow([f'error_{key}_{subkey}', subvalue])
                    else:
                        writer.writerow([f'error_{key}', value])
        else:
            # Use pandas for better CSV export
            data = []

            # Flatten the report data
            for key, value in report.summary.items():
                data.append({'category': 'summary', 'metric': key, 'value': value})

            for key, value in report.performance_metrics.items():
                data.append({'category': 'performance', 'metric': key, 'value': value})

            for key, value in report.error_analysis.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        data.append({'category': 'error', 'metric': f'{key}_{subkey}', 'value': subvalue})
                else:
                    data.append({'category': 'error', 'metric': key, 'value': value})

            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)

    def clear_data(self):
        """Clear all loaded monitoring data."""
        self.data.clear()
        self.llm_data.clear()
        self.neo4j_data.clear()
        self.loaded_files.clear()


def generate_monitoring_report(monitoring_dir: Union[Path, str],
                             output_path: Union[Path, str],
                             format: str = "html") -> bool:
    """
    Convenience function to generate a monitoring report.

    Args:
        monitoring_dir: Directory containing monitoring log files
        output_path: Output file path for the report
        format: Export format ("json", "html", "csv")

    Returns:
        True if successful, False otherwise
    """
    try:
        dashboard = MonitoringDashboard()
        records_loaded = dashboard.load_data(monitoring_dir)

        if records_loaded == 0:
            print(f"No monitoring data found in {monitoring_dir}")
            return False

        print(f"Loaded {records_loaded} monitoring records")

        report = dashboard.generate_comprehensive_report()
        success = dashboard.export_report(report, output_path, format)

        if success:
            print(f"Report generated: {output_path}")

        return success

    except Exception as e:
        print(f"Error generating report: {e}")
        return False
