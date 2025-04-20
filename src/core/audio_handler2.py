import sounddevice as sd
import numpy as np
# STT - Use whispercpp
from whispercpp import Whisper
# VAD
import webrtcvad
# TTS - Kokoro
from kokoro import KPipeline
import soundfile as sf
import os
import time
import collections
import torch # For Kokoro tensor check

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
        self.sample_rate_vad = self.audio_config['stt_sample_rate'] # VAD uses STT rate
        self.frame_length_vad = int(self.sample_rate_vad * (self.vad_frame_ms / 1000.0))
        self.padding_frames = int(self.vad_padding_ms / self.vad_frame_ms)

        # Temporary files (still useful for playback/debug)
        self.temp_output_file = "temp_ai_output.wav" # Keep for TTS output

        print("Initializing Audio Handler (Optimized)...")
        self._load_stt_model()
        self._load_tts_engine()
        self._check_audio_devices()
        print("Audio Handler initialized.")

    def _check_audio_devices(self): # Keep this
        try: sd.check_input_settings(); sd.check_output_settings(); print("Audio devices ok.")
        except Exception as e: print(f"Fatal: Audio device error: {e}"); raise

# Inside src/core/audio_handler.py

# --- UPDATED FUNCTION ---
    def _load_stt_model(self):
        # 1. Get the MODEL NAME from config, not the path
        model_identifier = self.stt_config['model_name']
        print(f"Loading STT model ({self.stt_config['provider']}: {model_identifier})...")

        if self.stt_config['provider'] == 'whisper.cpp':
            try:
                self.stt_model = Whisper.from_pretrained(model_identifier)
                print("Whisper.cpp STT model loaded.")
            except Exception as e:
                # 4. Update error message slightly
                print(f"Error loading Whisper.cpp model ('{model_identifier}'): {e}")
                print("Ensure 'whispercpp' is installed. It might download models on first use.")
                raise # Re-raise the exception to stop execution if loading fails
        else:
            raise ValueError(f"Unsupported STT provider: {self.stt_config['provider']}")
    # --- END OF UPDATED FUNCTION ---

    def _load_tts_engine(self): # Keep Kokoro loading
        print(f"Loading TTS engine ({self.tts_config['provider']})...")
        if self.tts_config['provider'] == 'kokoro':
            kokoro_config = self.tts_config.get('kokoro', {})
            repo_id = kokoro_config.get('repo_id')
            lang_code = kokoro_config.get('lang_code', 'a')
            try:
                self.tts_engine = KPipeline(repo_id=repo_id, lang_code=lang_code, device='cpu')
                self.default_kokoro_voice = kokoro_config.get('default_voice', 'en_speaker_1')
                print("Kokoro TTS engine loaded.")
            except Exception as e: print(f"Error loading Kokoro TTS: {e}"); raise
        else: raise ValueError(f"Unsupported TTS provider: {self.tts_config['provider']}")

    # --- NEW: Record with VAD ---
    def record_audio_with_vad(self):
        """Records audio using VAD to detect speech, returns audio buffer."""
        print("Listening... (Speak clearly, pause to finish)")
        frames_per_buffer = self.frame_length_vad
        padding_ring_buffer = collections.deque(maxlen=self.padding_frames)
        triggered = False
        speech_frames = []
        start_time = None
        last_speech_time = None

        stream = sd.InputStream(
            samplerate=self.sample_rate_vad,
            channels=1,
            dtype='int16',
            blocksize=frames_per_buffer # Process audio in VAD frame sizes
        )
        stream.start()

        while True:
            frame, overflowed = stream.read(frames_per_buffer)
            if overflowed:
                print("Warning: Input overflowed")
            frame_bytes = frame.tobytes()

            is_speech = False
            try:
                 is_speech = self.vad.is_speech(frame_bytes, self.sample_rate_vad)
            except Exception as e:
                 print(f"VAD processing error: {e}") # Catch potential errors

            timestamp = time.time()

            if not triggered:
                padding_ring_buffer.append((frame_bytes, is_speech))
                num_voiced = len([f for f, speech in padding_ring_buffer if speech])
                # Start recording when speech is detected
                if num_voiced > 0.8 * self.padding_frames: # Adjust threshold?
                    print("Speech detected, recording...")
                    triggered = True
                    start_time = timestamp
                    last_speech_time = timestamp
                    # Add buffered padding frames
                    for f_bytes, _ in padding_ring_buffer:
                        speech_frames.append(f_bytes)
                    padding_ring_buffer.clear()
            else:
                speech_frames.append(frame_bytes)
                if is_speech:
                    last_speech_time = timestamp
                # Stop recording if silence duration exceeds threshold
                silence_duration = timestamp - last_speech_time
                if silence_duration > self.vad_max_silence_s:
                    print("Silence detected, stopping recording.")
                    break

            # Timeout safeguard (optional)
            # if start_time and (timestamp - start_time > 30): # Max 30s recording
            #     print("Max recording time reached.")
            #     break

        stream.stop()
        stream.close()

        if not triggered or not speech_frames:
            print("No speech detected.")
            return None

        # Check minimum speech duration
        total_duration_s = len(speech_frames) * (self.vad_frame_ms / 1000.0)
        if total_duration_s < self.vad_min_speech_s:
            print(f"Recorded speech too short ({total_duration_s:.2f}s < {self.vad_min_speech_s:.2f}s). Ignoring.")
            return None

        # Combine frames and return as numpy array
        audio_data = np.frombuffer(b"".join(speech_frames), dtype=np.int16)
        print(f"Recording finished ({total_duration_s:.2f}s).")
        return audio_data # Return raw audio data

    # --- UPDATED: Transcribe using whispercpp ---
    def transcribe_audio(self, audio_data_np):
        """Transcribes numpy audio data using whisper.cpp."""
        print("Transcribing audio (whisper.cpp)...")
        if audio_data_np is None or audio_data_np.size == 0:
            return None
        try:
            # whispercpp expects float32 data normalized between -1 and 1
            audio_float32 = audio_data_np.astype(np.float32) / 32768.0

            # Transcribe using whispercpp
            # Adjust params as needed (e.g., n_threads)
            result = self.stt_model.transcribe(
                audio_float32,
                lang=self.stt_config.get('language', 'en'),
                # Other params like 'n_threads': self.stt_config.get('n_threads')
            )
            transcription = result.strip()
            print(f"Transcription: {transcription}")
            # Add filtering if needed
            return transcription
        except Exception as e:
            print(f"Error during whisper.cpp transcription: {e}")
            return None

    # --- UPDATED: Synthesize Speech (remains Kokoro) ---
    def synthesize_speech(self, text, output_filename=None):
        # ... (Keep Kokoro synthesize_speech function, ensure it returns output_filename or None) ...
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


    # --- UPDATED: Playback (remains soundfile) ---
    def play_audio(self, filename): # Keep this
        # ... (Keep soundfile playback function) ...
        print("Playing response...")
        if not filename or not os.path.exists(filename) or os.path.getsize(filename) == 0: return False
        try:
            audio_data, samplerate = sf.read(filename, dtype='float32')
            sd.play(audio_data, samplerate=samplerate, blocking=True); sd.stop()
            print("Playback finished."); return True
        except Exception as e: print(f"Error playing audio: {e}"); return False

    # --- UPDATED: Speak (uses updated synthesize) ---
    def speak(self, text): # Keep this structure
        # ... (Keep speak function, it calls updated synthesize/play) ...
        print(f"AI Speaking: {text}")
        temp_speak_file = f"temp_speak_{int(time.time() * 1000)}.wav"
        output_path = self.synthesize_speech(text, temp_speak_file)
        playback_success = False
        if output_path: playback_success = self.play_audio(output_path)
        else: print(f"ERROR: Could not synthesize speech for: '{text}'")
        try: # Cleanup
            if os.path.exists(temp_speak_file): os.remove(temp_speak_file)
        except Exception as e: print(f"Warning: Failed to remove temp speak file: {e}")
        return playback_success


    # --- UPDATED: Cleanup ---
    def cleanup_temp_files(self): # Keep only output file cleanup
        # We no longer create a temp input file
        files_to_remove = [self.temp_output_file]
        for f in files_to_remove:
            try:
                if os.path.exists(f): os.remove(f)
            except OSError as e: print(f"Warning: Error cleaning up '{f}': {e}")
