"""
Oceanus — One-command launcher.
Generates demo data then starts the 3D mission control server.

Usage:
    python run.py              # starts on port 8001
    python run.py --port 9000  # custom port
    python run.py --no-record  # skip re-recording episodes
"""
import argparse
import os
import sys
import subprocess
import webbrowser
import time

def main():
    parser = argparse.ArgumentParser(description="Oceanus Mission Control Launcher")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-record", action="store_true", help="Skip episode recording")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  🌊  OCEANUS MISSION CONTROL")
    print("  Multi-Agent RL Environment — 3D Demo Server")
    print("="*60)

    # Step 1: Generate demo data
    if not args.no_record:
        baseline_exists = os.path.exists("data/baseline_episode.json")
        trained_exists  = os.path.exists("data/trained_episode.json")
        if not baseline_exists or not trained_exists:
            print("\n[1/2] Generating demo episodes...")
            result = subprocess.run([sys.executable, "oceanus/demo_recorder.py"], check=True)
        else:
            print("\n[1/2] Demo data already exists. Use --no-record to skip or delete data/ to regenerate.")
    else:
        print("\n[1/2] Skipping episode recording.")

    # Step 2: Patch the HTML with the correct port
    html_path = "dashboard/index.html"
    if os.path.exists(html_path):
        with open(html_path, encoding="utf-8") as f:
            html = f.read()
        # Update port in WebSocket URL
        import re
        html = re.sub(
            r"const WS_PORT = location\.port \|\| [0-9]+;",
            f"const WS_PORT = location.port || {args.port};",
            html
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

    # Step 3: Start server
    print(f"\n[2/2] Starting 3D Mission Control on http://localhost:{args.port}")
    print(f"      Open your browser to: http://localhost:{args.port}")
    print(f"\n      Press Ctrl+C to stop.\n")

    # Auto-open browser after 2 seconds
    def open_browser():
        time.sleep(2)
        webbrowser.open(f"http://localhost:{args.port}")

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Start uvicorn
    os.execv(sys.executable, [
        sys.executable, "-m", "uvicorn",
        "dashboard.server:app",
        "--host", args.host,
        "--port", str(args.port),
        "--reload"
    ])

if __name__ == "__main__":
    main()
