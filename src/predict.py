"""
This file contains the Predictor class, which is used to run predictions on the
Whisper model. It is based on the Predictor class from the original Whisper
repository, with some modifications to make it work with the RP platform.
"""

from concurrent.futures import ThreadPoolExecutor
import numpy as np

from runpod.serverless.utils import rp_cuda

from faster_whisper import WhisperModel
from faster_whisper.utils import format_timestamp
from pyannote.audio import Pipeline
import torch


class Predictor:
    """ A Predictor class for the Whisper model """

    def __init__(self):
        self.models = {}
        self.diarization_pipeline = None

    def load_model(self, model_name):
        """ Load the model from the weights folder. """
        loaded_model = WhisperModel(
            model_name,
            device="cuda" if rp_cuda.is_available() else "cpu",
            compute_type="float16" if rp_cuda.is_available() else "int8")

        return model_name, loaded_model

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        model_names = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
        with ThreadPoolExecutor() as executor:
            for model_name, model in executor.map(self.load_model, model_names):
                if model_name is not None:
                    self.models[model_name] = model

        self.diarization_pipeline = None

    def load_diarization(self, hf_token):
        """Load the diarization pipeline with authentication"""
        if self.diarization_pipeline is None:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            if torch.cuda.is_available():
                self.diarization_pipeline = self.diarization_pipeline.to(torch.device("cuda"))
    
    def get_diarization(self, audio_path, hf_token):
        """Get speaker diarization for audio file"""
        self.load_diarization(hf_token)
        diarization = self.diarization_pipeline(audio_path)
        
        diarized_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            diarized_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        return diarized_segments

    def align_words_with_speakers(self, word_timestamps, diarized_segments):
        """Align word timestamps with speaker segments"""
        words_with_speakers = []
        
        for word in word_timestamps:
            word_mid_time = (word["start"] + word["end"]) / 2
            speaker = None
            
            for segment in diarized_segments:
                if segment["start"] <= word_mid_time <= segment["end"]:
                    speaker = segment["speaker"]
                    break
            
            words_with_speakers.append({
                "word": word["word"],
                "start": word["start"],
                "end": word["end"],
                "speaker": speaker
            })
            
        return words_with_speakers

    def predict(
        self,
        audio,
        model_name="base",
        transcription="plain_text",
        translate=False,
        translation="plain_text",
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False,
        enable_diarization=False,
        hf_token=None
    ):
        """Run a single prediction on the model"""

        model = self.models.get(model_name)
        if not model:
            raise ValueError(f"Model '{model_name}' not found.")

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        segments, info = model.transcribe(str(audio),
                                          language=language,
                                          task="transcribe",
                                          beam_size=beam_size,
                                          best_of=best_of,
                                          patience=patience,
                                          length_penalty=length_penalty,
                                          temperature=temperature,
                                          compression_ratio_threshold=compression_ratio_threshold,
                                          log_prob_threshold=logprob_threshold,
                                          no_speech_threshold=no_speech_threshold,
                                          condition_on_previous_text=condition_on_previous_text,
                                          initial_prompt=initial_prompt,
                                          prefix=None,
                                          suppress_blank=True,
                                          suppress_tokens=[-1],
                                          without_timestamps=False,
                                          max_initial_timestamp=1.0,
                                          word_timestamps=True,  # Always get word timestamps
                                          vad_filter=enable_vad
                                          )

        segments = list(segments)

        if enable_diarization:
            if not hf_token:
                raise ValueError("HuggingFace token (hf_token) is required for diarization")
            
            diarized_segments = self.get_diarization(audio, hf_token)
            
            # Integrate diarization information into segments
            for segment in segments:
                segment.words = self.align_words_with_speakers(segment.words, diarized_segments)
        
        # Process segments
        processed_segments = []
        speaker_transcript = []
        full_transcript = []

        for segment in segments:
            processed_segment = {
                "id": segment.id,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
                "words": segment.words
            }
            processed_segments.append(processed_segment)
            full_transcript.append(segment.text)

            if enable_diarization:
                current_speaker = None
                current_text = []
                for word in segment.words:
                    if word["speaker"] != current_speaker:
                        if current_text:
                            speaker_transcript.append({
                                "speaker": current_speaker,
                                "text": " ".join(current_text)
                            })
                            current_text = []
                        current_speaker = word["speaker"]
                    current_text.append(word["word"])
                if current_text:
                    speaker_transcript.append({
                        "speaker": current_speaker,
                        "text": " ".join(current_text)
                    })

        results = {
            "detected_language": info.language,
            "transcription": " ".join(full_transcript),
            "segments": processed_segments,
            "device": "cuda" if rp_cuda.is_available() else "cpu",
            "model": model_name,
        }

        if enable_diarization:
            results["speaker_transcript"] = speaker_transcript

        if translate:
            translation_segments, translation_info = model.transcribe(
                str(audio),
                task="translate",
                temperature=temperature
            )
            results["translation"] = format_segments(translation, translation_segments)

        return results

def serialize_segments(transcript):
    '''
    Serialize the segments to be returned in the API response.
    '''
    return [{
        "id": segment.id,
        "seek": segment.seek,
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "tokens": segment.tokens,
        "temperature": segment.temperature,
        "avg_logprob": segment.avg_logprob,
        "compression_ratio": segment.compression_ratio,
        "no_speech_prob": segment.no_speech_prob
    } for segment in transcript]


def format_segments(format, segments):
    '''
    Format the segments to the desired format
    '''

    if format == "plain_text":
        return " ".join([segment.text.lstrip() for segment in segments])
    elif format == "formatted_text":
        return "\n".join([segment.text.lstrip() for segment in segments])
    elif format == "srt":
        return write_srt(segments)
    
    return write_vtt(segments)


def write_vtt(transcript):
    '''
    Write the transcript in VTT format.
    '''
    result = ""

    for segment in transcript:
        result += f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result


def write_srt(transcript):
    '''
    Write the transcript in SRT format.
    '''
    result = ""

    for i, segment in enumerate(transcript, start=1):
        result += f"{i}\n"
        result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
        result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
        result += f"{segment.text.strip().replace('-->', '->')}\n"
        result += "\n"

    return result
