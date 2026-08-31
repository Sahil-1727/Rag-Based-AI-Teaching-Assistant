from faster_whisper import WhisperModel
import time

# Measure model loading time
start_load = time.time()

model = WhisperModel(
    "large-v2",
    device="cpu",
    compute_type="int8"
)

end_load = time.time()

# Measure transcription time
start_transcribe = time.time()

segments, info = model.transcribe(
    "audios/sample.mp3",
    language="hi",
    task="translate",
    word_timestamps=False
)

# Important: segments is a generator
# Actual transcription happens when we iterate over it
text = ""

for segment in segments:
    text += segment.text

end_transcribe = time.time()

print("\n--- Faster Whisper Results ---")
print(text)

print(f"\nModel loading time: {end_load - start_load:.2f} seconds")
print(f"Transcription time: {end_transcribe - start_transcribe:.2f} seconds")
print(f"Total time: {end_transcribe - start_load:.2f} seconds")