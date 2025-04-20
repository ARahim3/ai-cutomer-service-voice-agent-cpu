import sounddevice as sd
import numpy as np
import webrtcvad # VAD
import soundfile as sf # For TTS output/playback
import os
import time
import collections
import torch # Needed by transformers/kokoro

# --- STT IMPORT (Insanely Fast Whisper / Transformers Pipeline) ---
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available # Optional check

# --- TTS IMPORT (Kokoro) ---
from kokoro import KPipeline

class AudioHandler:
    def __init__(self, config):
        self.config = config
        self.audio_config = config['audio']
        self.stt_config = config['stt']
        self.tts_config = config['tts']

        # VAD setup
        self.vad = webrtcvad.Vad(self.audio_config['vad_aggressiveness'])
        self.vad_frame_ms = self.audio_config['vad_frame_ms']
        self.vad_padding_ms = self.audio_config['vad_padding_ms']
        self.vad_min_speech_s = self.audio_config['vad_min_speech_ms'] / 1000.0
        self.vad_max_silence_s = self.audio_config['vad_max_silence_ms'] / 1000.0
        self.sample_rate_vad = self.audio_config['stt_sample_rate'] # VAD uses STT rate (16k typical)
        self.frame_length_vad = int(self.sample_rate_vad * (self.vad_frame_ms / 1000.0))
        self.padding_frames = int(self.vad_padding_ms / self.vad_frame_ms)

        # Temporary file for TTS output
        self.temp_output_file = "temp_ai_output.wav"

        print("Initializing Audio Handler (Optimized - InsanelyFastWhisper)...")
        self._load_stt_model()
        self._load_tts_engine()
        self._check_audio_devices()
        print("Audio Handler initialized.")

    def _check_audio_devices(self):
        """Checks if audio input/output devices are available."""
        try:
            print("Checking audio devices...")
            sd.check_input_settings()
            sd.check_output_settings()
            print("Audio devices seem ok.")
        except Exception as e:
            print(f"Fatal: Audio device error: {e}")
            raise # Stop initialization if devices fail

    def _load_stt_model(self):
        """Loads the STT model using Hugging Face pipeline."""
        model_id = self.stt_config['model_name'] # e.g., "openai/whisper-base.en"
        provider = self.stt_config['provider'] # e.g., "insanely-fast-whisper"
        print(f"Loading STT model ({provider}: {model_id})...")

        if provider == 'insanely-fast-whisper':
            try:
                # Determine device and dtype
                device = 0 if torch.cuda.is_available() else -1
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                attn_implementation = None # Default unless flash attention is available/suitable

                # Optional: Check for Flash Attention 2 support
                # Requires specific hardware (Ampere+) and library versions
                # if is_flash_attn_2_available() and torch.cuda.is_available():
                #     attn_implementation = "flash_attention_2"
                #     print("Flash Attention 2 available and will be used.")
                # else:
                #     # Use 'sdpa' (Scaled Dot Product Attention) if Flash Attn not available
                #     # Requires PyTorch 2.0+
                #     if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                #         attn_implementation = "sdpa"
                #         print("Scaled Dot Product Attention (SDPA) will be used.")


                # Create the pipeline
                self.stt_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    torch_dtype=torch_dtype,
                    device=device,
                    # Add model_kwargs if attention implementation determined
                    # model_kwargs = {"attn_implementation": attn_implementation} if attn_implementation else None
                    # Note: Ensure ONNX Runtime (onnxruntime/onnxruntime-gpu) is installed
                    # for potential automatic CPU/GPU acceleration by the pipeline/Optimum.
                )
                print(f"Insanely-Fast Whisper STT pipeline loaded (Model: {model_id}, Device: {'GPU' if device==0 else 'CPU'}).")

            except Exception as e:
                print(f"Error loading Insanely-Fast Whisper pipeline: {e}")
                print("Ensure 'transformers', 'torch', 'optimum', 'onnxruntime' (or -gpu) might be needed.")
                raise
        else:
             raise ValueError(f"Unsupported STT provider in config: {provider}")

    def _load_tts_engine(self):
        """Loads the Kokoro TTS engine."""
        print(f"Loading TTS engine ({self.tts_config['provider']})...")
        if self.tts_config['provider'] == 'kokoro':
            kokoro_config = self.tts_config.get('kokoro', {})
            repo_id = kokoro_config.get('repo_id') # Optional repo ID
            lang_code = kokoro_config.get('lang_code', 'a') # Default 'a'
            try:
                self.tts_engine = KPipeline(
                    repo_id=repo_id, # Defaults to hexgrad/Kokoro-82M if None
                    lang_code=lang_code,
                    device='cpu' # Explicitly CPU for TTS usually
                )
                # Store default voice for convenience
                self.default_kokoro_voice = kokoro_config.get('default_voice', 'en_speaker_1') # Make sure this voice exists
                print("Kokoro TTS engine loaded.")
            except Exception as e:
                print(f"Error loading Kokoro TTS model: {e}")
                print("Ensure 'kokoro', 'soundfile', and 'espeak-ng' (system) are installed.")
                raise
        else:
            raise ValueError(f"Unsupported TTS provider in config: {self.tts_config['provider']}")

    def record_audio_with_vad(self):
        """Records audio using VAD, returns audio data as numpy array or None."""
        print("Listening... (Speak clearly, pause to finish)")
        frames_per_buffer = self.frame_length_vad
        padding_ring_buffer = collections.deque(maxlen=self.padding_frames)
        triggered = False
        speech_frames = []
        start_time = None
        last_speech_time = time.time() # Initialize last speech time

        stream = sd.InputStream(
            samplerate=self.sample_rate_vad,
            channels=1,
            dtype='int16',
            blocksize=frames_per_buffer
        )
        stream.start()

        try:
            while True:
                frame, overflowed = stream.read(frames_per_buffer)
                if overflowed: print("Warning: Input overflowed")
                frame_bytes = frame.tobytes()

                is_speech = False
                try:
                     # Ensure frame has correct length for VAD
                     if len(frame_bytes) == frames_per_buffer * 2: # (int16 = 2 bytes)
                         is_speech = self.vad.is_speech(frame_bytes, self.sample_rate_vad)
                     else:
                          print(f"Warning: Incorrect frame size {len(frame_bytes)} for VAD.")
                          # Decide how to handle - skip frame? break?
                          continue
                except Exception as e: print(f"VAD processing error: {e}")

                timestamp = time.time()

                if not triggered:
                    padding_ring_buffer.append((frame_bytes, is_speech))
                    num_voiced = len([f for f, speech in padding_ring_buffer if speech])
                    if num_voiced > 0.8 * self.padding_frames: # Speech detected trigger
                        print("Speech detected, recording...")
                        triggered = True
                        start_time = timestamp
                        last_speech_time = timestamp
                        for f_bytes, _ in padding_ring_buffer: speech_frames.append(f_bytes)
                        padding_ring_buffer.clear()
                else: # Already triggered, collecting speech
                    speech_frames.append(frame_bytes)
                    if is_speech:
                        last_speech_time = timestamp
                    # Check for end of speech (silence duration exceeded)
                    silence_duration = timestamp - last_speech_time
                    if silence_duration > self.vad_max_silence_s:
                        print("Silence detected, stopping recording.")
                        break # Exit loop on silence

                # Optional timeout
                # if start_time and (timestamp - start_time > 30): break

        finally: # Ensure stream is always stopped and closed
            stream.stop()
            stream.close()

        if not triggered or not speech_frames:
            print("No speech detected or frames collected.")
            return None

        total_duration_s = len(speech_frames) * (self.vad_frame_ms / 1000.0)
        if total_duration_s < self.vad_min_speech_s:
            print(f"Recorded speech too short ({total_duration_s:.2f}s). Ignoring.")
            return None

        audio_data = np.frombuffer(b"".join(speech_frames), dtype=np.int16)
        print(f"Recording finished ({total_duration_s:.2f}s).")
        return audio_data # Return raw audio data as numpy array

    def transcribe_audio(self, audio_data_np):
        """Transcribes numpy audio data using the loaded STT pipeline."""
        print("Transcribing audio (insanely-fast-whisper)...")
        if audio_data_np is None or audio_data_np.size == 0:
            print("Warning: No audio data provided for transcription.")
            return None
        try:
            # Pipeline expects float32 numpy array
            audio_float32 = audio_data_np.astype(np.float32) / 32768.0 # Normalize to [-1, 1]
            input_sample_rate = self.audio_config['stt_sample_rate']

            # Call the HF pipeline
            outputs = self.stt_pipeline(
                {"array": audio_float32, "sampling_rate": input_sample_rate},
                chunk_length_s=30, # Standard Whisper chunk length
                batch_size=self.stt_config.get("batch_size", 4), # Batching for potential speedup
                return_timestamps=False, # We don't need timestamps now
            )

            transcription = outputs["text"].strip()
            print(f"Transcription: {transcription}")
            return transcription

        except Exception as e:
            print(f"Error during insanely-fast-whisper transcription: {e}")
            return None

    def synthesize_speech(self, text, output_filename=None):
        """Synthesizes speech from text using Kokoro TTS."""
        if output_filename is None: output_filename = self.temp_output_file
        print(f"Synthesizing speech ({self.tts_config['provider']})...")
        if not text: print("Warning: Empty text passed to synth."); return None
        success = False
        if self.tts_config['provider'] == 'kokoro':
            try:
                voice_id = self.default_kokoro_voice
                audio_chunks = []
                generator = self.tts_engine(text, voice=voice_id)
                for i, (gs, ps, audio_chunk) in enumerate(generator):
                    if audio_chunk is not None:
                        if isinstance(audio_chunk, torch.Tensor): audio_chunk = audio_chunk.cpu().numpy()
                        audio_chunks.append(audio_chunk)
                if audio_chunks:
                    combined_audio = np.concatenate(audio_chunks)
                    sf.write(output_filename, combined_audio, samplerate=self.audio_config['tts_sample_rate'])
                    success = True
                else: print("Error: Kokoro TTS yielded no audio.")
            except Exception as e: print(f"Error during Kokoro synthesis: {e}")
        # Add other providers if needed
        if success and os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
            print(f"Speech synthesized to {output_filename}")
            return output_filename
        else:
            print(f"Warning: TTS synthesis failed or empty file '{output_filename}'.")
            if os.path.exists(output_filename):
                try: os.remove(output_filename); 
                except: pass
            return None

    def play_audio(self, filename):
        """Plays back an audio file using soundfile and sounddevice."""
        print("Playing response...")
        if not filename or not os.path.exists(filename) or os.path.getsize(filename) == 0:
            print(f"Error: Audio file '{filename}' invalid for playback.")
            return False
        try:
            audio_data, samplerate = sf.read(filename, dtype='float32')
            sd.play(audio_data, samplerate=samplerate, blocking=True)
            sd.stop()
            print("Playback finished.")
            return True
        except Exception as e:
            print(f"Error playing audio from {filename}: {e}")
            return False

    def speak(self, text):
        """Convenience function to synthesize and play text immediately."""
        print(f"AI Speaking: {text}")
        # Use a unique filename based on time to avoid conflicts
        temp_speak_file = f"temp_speak_{int(time.time() * 1000)}.wav"
        output_path = self.synthesize_speech(text, temp_speak_file)
        playback_success = False
        if output_path:
            playback_success = self.play_audio(output_path)
            if not playback_success:
                 print(f"ERROR: Failed to play synthesized audio for: '{text}'")
        else:
            print(f"ERROR: Could not synthesize speech for: '{text}'")
        # Clean up the temporary file used for speaking
        try:
            if os.path.exists(temp_speak_file): os.remove(temp_speak_file)
        except Exception as e: print(f"Warning: Failed to remove temp speak file {temp_speak_file}: {e}")
        return playback_success

    def cleanup_temp_files(self):
        """Cleans up temporary output audio files."""
        # Input files are no longer saved, only output files might exist
        files_to_remove = [self.temp_output_file] # Add any other temp files if created
        for f in files_to_remove:
            try:
                if os.path.exists(f): os.remove(f)
            except OSError as e:
                print(f"Warning: Error cleaning up temp audio file '{f}': {e}")