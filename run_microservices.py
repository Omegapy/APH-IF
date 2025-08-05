#!/usr/bin/env python3
"""
APH-IF Microservices Development Runner
Advanced Parallel Hybrid - Intelligent Fusion

Development script to run backend and frontend microservices locally
for testing and development purposes.

Author: Alexander Ricciardi
Date: 2025-08-05
License: Apache-2.0
"""

import subprocess
import sys
import time
import signal
import os
from pathlib import Path
from typing import List, Optional

class MicroserviceRunner:
    """Manages running multiple microservices for development"""
    
    def __init__(self):
        self.processes: List[subprocess.Popen] = []
        self.project_root = Path(__file__).parent
        
    def run_backend(self, port: int = 8000) -> subprocess.Popen:
        """Run the FastAPI backend service"""
        print(f"🔧 Starting APH-IF Backend on port {port}...")
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "backend.main:app",
            "--host", "0.0.0.0",
            "--port", str(port),
            "--reload",
            "--log-level", "info"
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        
        process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(process)
        return process
    
    def run_frontend(self, port: int = 8501) -> subprocess.Popen:
        """Run the Streamlit frontend service"""
        print(f"🖥️ Starting APH-IF Frontend on port {port}...")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "app.py",
            "--server.address", "0.0.0.0",
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        
        process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(process)
        return process
    
    def run_bot(self, port: int = 8502) -> subprocess.Popen:
        """Run the Streamlit bot interface"""
        print(f"🤖 Starting APH-IF Bot on port {port}...")
        
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            "frontend/bot.py",
            "--server.address", "0.0.0.0",
            "--server.port", str(port),
            "--server.headless", "true"
        ]
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.project_root)
        
        process = subprocess.Popen(
            cmd,
            cwd=self.project_root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        self.processes.append(process)
        return process
    
    def wait_for_service(self, port: int, timeout: int = 30) -> bool:
        """Wait for a service to become available"""
        import requests
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}", timeout=1)
                if response.status_code < 500:
                    return True
            except:
                pass
            time.sleep(1)
        
        return False
    
    def run_all(self):
        """Run all microservices"""
        print("🚀 Starting APH-IF Microservices...")
        print("=" * 50)
        
        # Start backend first
        backend_process = self.run_backend(8000)
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        if self.wait_for_service(8000):
            print("✅ Backend is ready!")
        else:
            print("❌ Backend failed to start")
            return
        
        # Start frontend
        frontend_process = self.run_frontend(8501)
        
        # Start bot interface
        bot_process = self.run_bot(8502)
        
        # Wait for frontend services
        print("⏳ Waiting for frontend services...")
        time.sleep(5)  # Give Streamlit time to start
        
        print("\n🎉 All services started!")
        print("=" * 50)
        print("📋 Service URLs:")
        print("   🔧 Backend API:    http://localhost:8000")
        print("   📖 API Docs:       http://localhost:8000/docs")
        print("   🖥️ Frontend:       http://localhost:8501")
        print("   🤖 Bot Interface:  http://localhost:8502")
        print("   🗄️ Neo4j Browser:  http://localhost:7474")
        print("=" * 50)
        print("Press Ctrl+C to stop all services")
        
        # Monitor processes
        try:
            while True:
                # Check if any process has died
                for i, process in enumerate(self.processes):
                    if process.poll() is not None:
                        print(f"⚠️ Process {i} has stopped")
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping all services...")
            self.stop_all()
    
    def stop_all(self):
        """Stop all running processes"""
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except:
                pass
        
        self.processes.clear()
        print("✅ All services stopped")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="APH-IF Microservices Runner")
    parser.add_argument("--service", choices=["backend", "frontend", "bot", "all"], 
                       default="all", help="Service to run")
    parser.add_argument("--backend-port", type=int, default=8000, 
                       help="Backend port")
    parser.add_argument("--frontend-port", type=int, default=8501, 
                       help="Frontend port")
    parser.add_argument("--bot-port", type=int, default=8502, 
                       help="Bot port")
    
    args = parser.parse_args()
    
    runner = MicroserviceRunner()
    
    # Setup signal handler for graceful shutdown
    def signal_handler(sig, frame):
        print("\n🛑 Received interrupt signal...")
        runner.stop_all()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        if args.service == "backend":
            runner.run_backend(args.backend_port)
            print(f"🔧 Backend running on http://localhost:{args.backend_port}")
            
        elif args.service == "frontend":
            runner.run_frontend(args.frontend_port)
            print(f"🖥️ Frontend running on http://localhost:{args.frontend_port}")
            
        elif args.service == "bot":
            runner.run_bot(args.bot_port)
            print(f"🤖 Bot running on http://localhost:{args.bot_port}")
            
        else:
            runner.run_all()
            return
        
        # Wait for single service
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping service...")
            runner.stop_all()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        runner.stop_all()
        sys.exit(1)

if __name__ == "__main__":
    main()
