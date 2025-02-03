from transformers import pipeline
import torch
import config

device = torch.device("cpu")

generator = pipeline("text-generation", model=config.CHECKPOINT_PATH, device=device.index)

def generate_code(prompt):
    """Generates code completions using a CPU-only setup."""
    generated_text = generator(prompt, max_length=100, num_return_sequences=1)
    return generated_text[0]["generated_text"]

# CLI-based input for inference
if __name__ == "__main__":
    while True:
        prompt = input("\n🔹 Enter code prompt (or type 'exit' to quit): ")
        if prompt.lower() == "exit":
            break
        print("\n💡 Generated Code:\n", generate_code(prompt))