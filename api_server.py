from flask import Flask, request, jsonify
from transformers import pipeline
import config

app = Flask(__name__)

generator = pipeline("text-generation", model=config.CHECKPOINT_PATH)

@app.route('/generate', methods=['POST'])
def generate_code():
    data = request.get_json()
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 100)

    generated_text = generator(prompt, max_length=max_length, num_return_sequences=1)
    return jsonify(generated_text)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.API_PORT)