import subprocess
import time
import sys
import os

def start_service(name, path, port):
    print(f"[INFO] Starting {name} on port {port}...")
    # Use the current python executable to ensure we use the same environment
    cmd = [
        sys.executable, "-m", "uvicorn", "server:app",
        "--host", "0.0.0.0",
        "--port", str(port)
    ]
    return subprocess.Popen(cmd, cwd=path, stdout=sys.stdout, stderr=sys.stderr)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    services_dir = os.path.join(base_dir, "services")
    
    services = [
        ("Detection Engine", os.path.join(services_dir, "detection-engine"), 8000),
        ("Trainer Service", os.path.join(services_dir, "trainer"), 8001),
        ("Evaluation Service", os.path.join(services_dir, "eval-service"), 8002),
    ]
    
    processes = []
    
    try:
        for name, path, port in services:
            # Install requirements for each service first
            print(f"[INFO] Installing dependencies for {name}...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], cwd=path, check=True)
            
            p = start_service(name, path, port)
            processes.append(p)
            # Give it a moment to start
            time.sleep(2)
        
        print("\n[SUCCESS] All 3 services are running!")
        print("- Detection Engine: http://localhost:8000")
        print("- Trainer Service: http://localhost:8001")
        print("- Evaluation Service: http://localhost:8002")
        print("\nPress Ctrl+C to stop all services.")
        
        while True:
            time.sleep(1)
            # Check if any process has died
            for i, p in enumerate(processes):
                if p.poll() is not None:
                    print(f"[ERROR] Service {services[i][0]} has stopped unexpectedly.")
                    sys.exit(1)
                    
    except KeyboardInterrupt:
        print("\n[INFO] Stopping all services...")
        for p in processes:
            p.terminate()
        print("[INFO] All services stopped.")
    except Exception as e:
        print(f"[ERROR] {e}")
        for p in processes:
            p.terminate()
        sys.exit(1)

if __name__ == "__main__":
    main()
