#!/usr/bin/env python3
"""
Setup script for CM_PrivLaw Exam Assistant

This script will:
1. Check Python version
2. Install required packages
3. Verify FAISS index exists
"""

import sys
import subprocess
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def install_requirements():
    """Install required packages."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        sys.exit(1)
    
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ All packages installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        sys.exit(1)

def check_faiss_index():
    """Check if FAISS index exists."""
    index_path = Path("index") / "faiss_index"
    
    if index_path.exists():
        print(f"✅ FAISS index found at {index_path}")
    else:
        print(f"⚠️  FAISS index not found at {index_path}")
        print("   You need to build the index first with: python scripts/build_index.py")

def check_env_vars():
    """Check environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        print("✅ OPENAI_API_KEY is set")
    else:
        print("⚠️  OPENAI_API_KEY not set!")
        print("   Set it with: $env:OPENAI_API_KEY='your-key-here' (PowerShell)")
        print("   or: set OPENAI_API_KEY=your-key-here (CMD)")

def main():
    """Run setup."""
    print("🔧 Setting up CM_PrivLaw Exam Assistant...")
    print("=" * 50)
    
    check_python_version()
    install_requirements()
    check_faiss_index()
    check_env_vars()
    
    print("\n" + "=" * 50)
    print("🎉 Setup complete!")
    print("\nNext steps:")
    print("1. If FAISS index doesn't exist, run: python scripts/build_index.py")
    print("2. Make sure OPENAI_API_KEY is set in your environment")
    print("3. Start the GUI with: python run_gui.py")

if __name__ == "__main__":
    main()
