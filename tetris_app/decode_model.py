# decode_model.py
import base64

print("🔍 Reading base64 data...")
with open('static/models/tetris_dqn_agent.b64', 'r') as f:
    b64_data = f.read().replace('\n', '').strip()

print("🔄 Decoding and saving as .h5...")
with open('tetris_dqn_agent.h5', 'wb') as f:
    f.write(base64.b64decode(b64_data))

print("✅ Success! Model saved as 'tetris_dqn_agent.h5'")
