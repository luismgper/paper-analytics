"""Main entry points for the application."""
import subprocess
import sys
from pathlib import Path

def run_app():
    """Run the Streamlit app."""
    app_path = Path(__file__).parent / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])

def main():
    """Main entry point with CLI options."""
    import argparse
    
    parser = argparse.ArgumentParser(description="My Streamlit App")
    parser.add_argument("--port", type=int, default=8501, help="Port to run on")
    parser.add_argument("--host", default="localhost", help="Host to run on")
    
    args = parser.parse_args()
    
    app_path = Path(__file__).parent / "paper_analytics_dashboard.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app_path),
        "--server.port", str(args.port),
        "--server.address", args.host
    ]
    
    subprocess.run(cmd)

if __name__ == "__main__":
    main()