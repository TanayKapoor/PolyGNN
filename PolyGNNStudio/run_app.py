#!/usr/bin/env python3
"""
Wrapper script to run PolyGNN Showcase Streamlit app with proper configuration
to avoid torch.classes introspection issues.
"""

import os
import sys
import subprocess

def main():
    """Main function to set environment and run Streamlit app."""
    
    # Set environment variables to prevent torch.classes issues
    env_vars = {
        "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none",
        "STREAMLIT_SERVER_RUN_ON_SAVE": "false",
        "STREAMLIT_GLOBAL_SUPPRESS_DEPRECATION_WARNINGS": "true",
        "STREAMLIT_LOGGER_LEVEL": "warning",
        "TORCH_CLASSES_DISABLE": "1"  # Custom flag to disable torch classes introspection
    }
    
    # Update environment
    for key, value in env_vars.items():
        os.environ[key] = value
    
    # Print startup message
    print("🚀 Starting PolyGNN Showcase...")
    print("🔧 Configured to avoid torch.classes introspection issues")
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "app.py")
    
    # Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.port", "8502",
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--server.runOnSave", "false"
    ]
    
    try:
        # Run Streamlit
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 PolyGNN Showcase stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()