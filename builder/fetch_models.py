from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# Only load the large-v3 model
model_names = ["large-v3"]

def load_whisper_model(selected_model):
    '''
    Load and cache Whisper model
    '''
    for _attempt in range(5):
        while True:
            try:
                loaded_model = WhisperModel(
                    selected_model, device="cpu", compute_type="int8")
            except (AttributeError, OSError):
                continue

            break

    return selected_model, loaded_model

def load_pyannote_model():
    '''
    Load and cache pyannote model
    '''
    try:
        pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
    except Exception as e:
        print(f"Error loading pyannote model: {e}")
        return None

    return "pyannote", pipeline

models = {}

# Load Whisper model
with ThreadPoolExecutor() as executor:
    for model_name, model in executor.map(load_whisper_model, model_names):
        if model_name is not None:
            models[model_name] = model

# Load pyannote model
pyannote_model = load_pyannote_model()
if pyannote_model is not None:
    models[pyannote_model[0]] = pyannote_model[1]

print("Models loaded successfully:", models.keys())
