import keyboard
from transformers import pipeline
import config

# Load trained model for real-time code completion
generator = pipeline("text-generation", model=config.CHECKPOINT_PATH)

def generate_code(prompt):
    generated_text = generator(prompt, max_length=100, num_return_sequences=1)
    return generated_text[0]["generated_text"]

# Event listener for hotkey
def on_hotkey():
    prompt = input("\n🔹 Type code prompt: ")
    print("\n💡 Code Completion:\n", generate_code(prompt))

# Listen for hotkey
keyboard.add_hotkey(config.HOTKEY, on_hotkey)
print(f"\n🔥 Code Completion Active! Press {config.HOTKEY} anywhere to generate code.")
keyboard.wait()