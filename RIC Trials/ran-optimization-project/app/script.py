import requests
import time

EMULATOR_URL = "http://localhost:5000"

def check_emulator_health():
    try:
        response = requests.get(f"{EMULATOR_URL}/health")
        if response.status_code == 200:
            print("Emulator is healthy")
    except Exception as e:
        print(f"Failed to connect to emulator: {e}")

def send_optimization_data():
    payload = {
        "cell_id": "cell_1",
        "load": 85,
        "action": "reduce_load"
    }
    try:
        response = requests.post(f"{EMULATOR_URL}/cell_data", json=payload)
        print(f"Response from emulator: {response.json()}")
    except Exception as e:
        print(f"Error sending data: {e}")

if __name__ == "__main__":
    while True:
        check_emulator_health()
        send_optimization_data()
        time.sleep(10)  # Run every 10 seconds
