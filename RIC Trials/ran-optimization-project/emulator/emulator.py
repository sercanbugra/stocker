from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "emulator is running"}), 200

@app.route('/cell_data', methods=['POST'])
def receive_cell_data():
    data = request.json
    print(f"Received data: {data}")
    # Simulate RAN behavior
    return jsonify({"message": "Cell data processed"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
