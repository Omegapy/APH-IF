#!/usr/bin/env python3
"""
Test script for APH-IF Docker setup
Verifies that all components are working correctly
"""

import requests
import time
import sys
from pathlib import Path

def test_streamlit_health():
    """Test if Streamlit is running and healthy"""
    try:
        response = requests.get("http://localhost:8501/_stcore/health", timeout=10)
        if response.status_code == 200:
            print("✅ Streamlit is healthy")
            return True
        else:
            print(f"❌ Streamlit health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Streamlit connection failed: {e}")
        return False

def test_streamlit_app():
    """Test if Streamlit app is accessible"""
    try:
        response = requests.get("http://localhost:8501", timeout=10)
        if response.status_code == 200:
            print("✅ Streamlit app is accessible")
            return True
        else:
            print(f"❌ Streamlit app failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Streamlit app connection failed: {e}")
        return False

def test_neo4j_browser():
    """Test if Neo4j browser is accessible"""
    try:
        response = requests.get("http://localhost:7474", timeout=10)
        if response.status_code == 200:
            print("✅ Neo4j browser is accessible")
            return True
        else:
            print(f"❌ Neo4j browser failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Neo4j browser connection failed: {e}")
        return False

def test_file_structure():
    """Test if required files exist"""
    required_files = [
        "Dockerfile",
        "docker-compose.yml", 
        "app.py",
        "requirements.txt",
        ".streamlit/config.toml"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 Testing APH-IF Docker Setup")
    print("=" * 40)
    
    # Test file structure
    print("\n📁 Testing file structure...")
    files_ok = test_file_structure()
    
    # Test services (with retries)
    print("\n🌐 Testing services...")
    print("Waiting for services to start...")
    
    max_retries = 30
    retry_delay = 2
    
    for attempt in range(max_retries):
        print(f"Attempt {attempt + 1}/{max_retries}")
        
        streamlit_health = test_streamlit_health()
        streamlit_app = test_streamlit_app()
        neo4j_browser = test_neo4j_browser()
        
        if streamlit_health and streamlit_app and neo4j_browser:
            print("\n🎉 All tests passed!")
            print("\n📋 Service URLs:")
            print("   Streamlit App: http://localhost:8501")
            print("   Neo4j Browser: http://localhost:7474")
            print("   (Neo4j credentials: neo4j/YourStrongPassword)")
            return True
        
        if attempt < max_retries - 1:
            print(f"⏳ Retrying in {retry_delay} seconds...")
            time.sleep(retry_delay)
    
    print("\n❌ Some tests failed after maximum retries")
    print("\n🔧 Troubleshooting tips:")
    print("   1. Make sure Docker is running")
    print("   2. Check if ports 8501, 7474, 7687 are available")
    print("   3. Run: docker-compose logs -f")
    print("   4. Try: docker-compose down && docker-compose up --build")
    
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
