"""
Platform Launcher Script
========================
One-click execution script that launches the Customer Churn Platform
and automatically opens the browser.

Usage:
    python run.py
"""

import os
import sys
import subprocess
import webbrowser
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 65)
    print("🚀 CUSTOMER CHURN & REVENUE INTELLIGENCE PLATFORM")
    print("=" * 65)
    print(f"📁 Project Directory: {ROOT}")

    # Step 1: Check dependencies
    print("\n🔍 Checking core dependencies...")
    required = ["streamlit", "pandas", "numpy", "sklearn", "plotly"]
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✅ {pkg} is installed")
        except ImportError:
            missing.append(pkg)
            print(f"  ❌ {pkg} is missing")

    if missing:
        print(f"\n⚠️ Missing dependencies: {missing}")
        print("Installing missing packages now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])

    # Step 2: Open browser after 2 seconds
    url = "http://localhost:8501"
    print(f"\n🌐 Launching web browser to: {url}")
    print("📌 Keep this terminal window open while using the platform!")
    print("   Press Ctrl + C to stop the application.\n")

    def open_browser():
        time.sleep(2)
        webbrowser.open(url)

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Step 3: Run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        os.path.join(ROOT, "app.py"),
        "--server.headless=false",
        "--theme.base=dark"
    ]
    subprocess.run(cmd, cwd=ROOT)

if __name__ == "__main__":
    main()
