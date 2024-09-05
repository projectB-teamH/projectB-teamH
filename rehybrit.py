import pyaudio
import queue
import sys
import threading
import keyboard
import io
from google.cloud import speech_v1p1beta1 as speech
from google.cloud import translate_v2 as translate
import torch
from speechbrain.pretrained import SpeakerRecognition

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

# Load pre-trained speaker recognition model
speaker_recognition = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmp_model")

class MicrophoneStream(object):
    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        self.closed = False
        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break
            yield b"".join(data)

def translate_text(text, target_language):
    translate_client = translate.Client()
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

def save_transcript(transcript, filename):
    with open(filename, 'a', encoding='utf-8') as file:
        file.write(transcript + '\n')

def listen_print_loop(responses, filename, speaker_profiles):
    for response in responses:
        if not response.results:
            continue
        result = response.results[0]
        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript
        print(f"Transcript: {transcript}")

        # Speaker recognition
        matched_speaker = recognize_speaker('temp_audio.wav', speaker_profiles)
        speaker_name = f"[Speaker {matched_speaker}]"
        
        if is_japanese(transcript):
            translated_text = translate_text(transcript, 'en')
        else:
            translated_text = translate_text(transcript, 'ja')

        print(f"{speaker_name} Translated: {translated_text}")

        save_transcript(f"{speaker_name} Original: {transcript}\nTranslated: {translated_text}", filename)

def recognize_speaker(audio_path, speaker_profiles):
    """
    認識した音声を各プロファイルと比較し、最も一致する話者を特定する。
    """
    for idx, profile in enumerate(speaker_profiles, 1):
        result = speaker_recognition.verify_files(profile, audio_path)
        if result:
            return idx  # 話者番号を返す

    return "Unknown"  # 話者が特定できない場合

def is_japanese(text):
    for ch in text:
        if ord(ch) > 0x3000:
            return True
    return False

def check_for_space_key(stream):
    print("Press SPACE to stop the recording...")
    keyboard.wait('space')
    stream.closed = True
    print("Recording stopped.")

def real_time_recognition(filename, speaker_profiles):
    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="ja-JP",
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (speech.StreamingRecognizeRequest(audio_content=content) for content in audio_generator)
        responses = client.streaming_recognize(config=streaming_config, requests=requests)

        stop_thread = threading.Thread(target=check_for_space_key, args=(stream,))
        stop_thread.start()

        listen_print_loop(responses, filename, speaker_profiles)

def transcribe_audio(file_path, filename, speaker_profiles):
    client = speech.SpeechClient()

    with io.open(file_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=44100,
        language_code="ja-JP",
    )

    response = client.recognize(config=config, audio=audio)

    for result in response.results:
        transcript = result.alternatives[0].transcript
        print("Transcript: {}".format(transcript))

        # Speaker recognition
        matched_speaker = recognize_speaker(file_path, speaker_profiles)
        speaker_name = f"[Speaker {matched_speaker}]"
        
        save_transcript(f"{speaker_name} Original: {transcript}", filename)

def read_transcript(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            content = file.read()
            print("\n--- Saved Transcript ---")
            print(content)
            print("--- End of Transcript ---\n")
    except FileNotFoundError:
        print(f"No transcript found in '{filename}'.")

def main():
    filename = input("Enter the filename to save the transcript (e.g., transcript.txt): ")

    # 複数の話者プロフィールのパスを取得
    num_speakers = int(input("Enter the number of speakers to recognize: "))
    speaker_profiles = []
    for i in range(num_speakers):
        profile_path = input(f"Enter the path to the speaker {i + 1} profile audio file: ")
        speaker_profiles.append(profile_path)

    mode = input("Choose mode: (1) Real-time recognition (2) Transcribe audio file (3) Read saved transcript: ")

    if mode == '1':
        real_time_recognition(filename, speaker_profiles)
    elif mode == '2':
        file_path = input("Enter the path to the audio file: ")
        transcribe_audio(file_path, filename, speaker_profiles)
    elif mode == '3':
        read_transcript(filename)
    else:
        print("Invalid mode selected.")

if __name__ == "__main__":
    main()
