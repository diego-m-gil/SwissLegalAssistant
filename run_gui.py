#!/usr/bin/env python3
"""
Launch the CM_PrivLaw Exam Assistant GUI

Usage: python run_gui.py
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def main():
    """Launch the Streamlit GUI."""
    # Try to load API key from .env file
    env_path = Path(__file__).parent / ".env"
    template_env_path = Path(__file__).parent / ".env.template"
    
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print("✅ Loaded API key from .env file")
    elif template_env_path.exists():
        load_dotenv(dotenv_path=template_env_path)
        print("⚠️ Using API key from .env.template - for development only!")
    
    # Check for OpenAI API key - but let Streamlit handle missing keys
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️ No API key found in environment variables or .env file")
        print("You'll need to provide an API key in the Streamlit interface")
    
    # Get the scripts directory
    scripts_dir = Path(__file__).parent / "scripts"
    app_path = scripts_dir / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"ERROR: Streamlit app not found at {app_path}")
        sys.exit(1)
    
    print("🚀 Starting CM_PrivLaw Exam Assistant GUI...")
    print("🌐 The app will open in your default browser")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.headless", "false",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down the GUI...")
    except Exception as e:
        print(f"❌ Failed to start GUI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
