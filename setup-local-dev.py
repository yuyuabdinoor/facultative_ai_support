#!/usr/bin/env python3
"""
Setup script to configure local development environment
Detects your Python environment and generates the correct docker-compose configuration
"""

import sys
import os
import site
from pathlib import Path

def detect_python_environment():
    """Detect the current Python environment and site-packages location"""
    
    print("🔍 Detecting Python environment...")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    # Get site-packages directories
    site_packages = site.getsitepackages()
    user_site = site.getusersitepackages()
    
    print(f"\n📦 Site-packages locations:")
    for i, path in enumerate(site_packages):
        print(f"  {i+1}. {path}")
    
    if user_site:
        print(f"  User site: {user_site}")
    
    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    if in_venv:
        print(f"\n🐍 Virtual environment detected: {sys.prefix}")
        venv_site_packages = os.path.join(sys.prefix, 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
        recommended_path = venv_site_packages
    else:
        print(f"\n🐍 System Python detected")
        recommended_path = site_packages[0] if site_packages else user_site
    
    return recommended_path

def generate_docker_compose_override(site_packages_path):
    """Generate docker-compose.local.yml with the correct Python path"""
    
    template = f"""# Docker Compose override for local development
# Generated automatically - uses your local Python environment
version: '3.8'

services:
  # FastAPI Backend with local Python environment
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.local
    volumes:
      - ./backend:/app
      - ./uploads:/app/uploads
      # Your local Python site-packages
      - {site_packages_path}:/usr/local/lib/python3.12/site-packages:ro
    environment:
      - PYTHONPATH=/app:/usr/local/lib/python3.12/site-packages

  # Celery Worker with local Python environment
  celery_worker:
    build:
      context: ./backend
      dockerfile: Dockerfile.local
    volumes:
      - ./backend:/app
      - ./uploads:/app/uploads
      - {site_packages_path}:/usr/local/lib/python3.12/site-packages:ro
    environment:
      - PYTHONPATH=/app:/usr/local/lib/python3.12/site-packages
    command: python -m celery -A app.celery worker --loglevel=info

  # Celery Beat with local Python environment
  celery_beat:
    build:
      context: ./backend
      dockerfile: Dockerfile.local
    volumes:
      - ./backend:/app
      - {site_packages_path}:/usr/local/lib/python3.12/site-packages:ro
    environment:
      - PYTHONPATH=/app:/usr/local/lib/python3.12/site-packages
    command: python -m celery -A app.celery beat --loglevel=info
"""
    
    with open('docker-compose.local.yml', 'w') as f:
        f.write(template)
    
    print(f"✅ Generated docker-compose.local.yml with path: {site_packages_path}")

def check_required_packages():
    """Check if required packages are installed locally"""
    
    required_packages = [
        'fastapi', 'uvicorn', 'torch', 'transformers', 
        'sentence_transformers', 'doctr', 'celery', 'redis'
    ]
    
    print(f"\n📋 Checking required packages...")
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NOT FOUND")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print(f"Install them with: pip install {' '.join(missing_packages)}")
        return False
    else:
        print(f"\n🎉 All required packages are installed!")
        return True

def main():
    print("🚀 Setting up local development environment for AI Reinsurance System")
    print("=" * 70)
    
    # Check if venv exists in backend directory
    venv_path = "./backend/venv/lib/python3.12/site-packages"
    if os.path.exists(venv_path):
        print(f"✅ Found virtual environment at: {venv_path}")
        site_packages_path = venv_path
    else:
        print(f"⚠️  Virtual environment not found at: {venv_path}")
        print("   Detecting alternative Python environment...")
        site_packages_path = detect_python_environment()
    
    # Check required packages
    packages_ok = check_required_packages()
    
    # Generate docker-compose override (if needed)
    if site_packages_path != venv_path:
        generate_docker_compose_override(site_packages_path)
        print(f"\n📝 Generated custom docker-compose.local.yml for path: {site_packages_path}")
    else:
        print(f"\n✅ Using default configuration (venv in backend directory)")
    
    print("\n" + "=" * 70)
    print("🎯 Setup complete! To start development:")
    print("")
    print("1. Install missing packages (if any) in your venv:")
    print("   source backend/venv/bin/activate")
    print("   pip install -r backend/requirements.txt")
    print("")
    print("2. Start the development environment:")
    print("   docker-compose up")
    print("")
    print("3. Your local Python packages will be mounted into the containers!")
    print("   No need to reinstall PyTorch and other heavy packages 🎉")
    
    if not packages_ok:
        print("\n⚠️  Note: Install missing packages first before starting Docker")

if __name__ == "__main__":
    main()