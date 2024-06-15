#!/usr/bin/env python

import argparse
import io
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import speech_recognition as sr
import time
import tempfile
import re
import num2words

try:
    import winsound
    playsound = None
except ImportError:
    winsound = None
    from playsound import playsound

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from faster_whisper import WhisperModel



def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def rewrite(path, data):
    if not path:
        return

    with open(path, "w", encoding="utf-8") as f:
        print(f" *** {datetime.now()}: lines {len(data)}", file=f)
        f.write(os.linesep)
        #x = 0
        for l in data:
            f.write(l)
            f.write(os.linesep)


def remove_dots_between_numbers(input_string):
    # Define a regular expression pattern to match dots between numbers
    pattern = r'(\d+)\.(\d+)'

    # Define a function to replace dots with an empty string for matched patterns
    def replace_dots(match):
        return match.group(1) + match.group(2)

    # Use re.sub() to apply the replacement function to the input string
    result_string = re.sub(pattern, replace_dots, input_string)

    return result_string


def convert_num_to_words(input_text, lang):
    utterance = remove_dots_between_numbers(input_text)
    utterance = ' '.join([num2words.num2words(i, lang=lang) if i.isdigit() else i for i in utterance.split()])
    return utterance


def init_tts(args):
        # Late-import to make things load faster
        from TTS.api import TTS
        from TTS.utils.manage import ModelManager
        from TTS.utils.synthesizer import Synthesizer

        tts = TTS(model_name=args.tts_model_name, progress_bar=True).to(args.device)

        return tts

        ## load models
        #synthesizer = Synthesizer(
        #    model_name=args.tts_model_name,
        #    speaker_wav=args.speaker_wav,
        #    model_path=args.tts_model_path,
        #    model_config=args.tts_model_config,
        #    language_name=args.language
        #).to(args.device)

        #return synthesizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="large-v2", help="Model to use",
                        choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"])
    parser.add_argument("--non_english", action='store_true', default=True,
                        help="Don't use the english model.")
    parser.add_argument("--energy_threshold", default=1000,
                        help="Energy level for mic to detect.", type=int)
    parser.add_argument("--record_timeout", default=2,
                        help="How real time the recording is in seconds.", type=float)
    parser.add_argument("--phrase_timeout", default=3,
                        help="How much empty space between recordings before we "
                             "consider it a new line in the transcription.", type=float)  
    parser.add_argument("--tts_model_name", default=None,
                        help="TTS model name", type=str)
    parser.add_argument("--tts_model_path", default=None,
                        help="Path to TTS model", type=str)
    parser.add_argument("--tts_model_config", default=None,
                        help="Path to TTS config", type=str)
    parser.add_argument("--language", default=None,
                        help="Language", type=str)

    parser.add_argument("--output_file", default=None,
                        help="Output text to file.", type=str)
    parser.add_argument("--device", default="cuda",
                        help="CUDA or cpu", type=str)
    parser.add_argument(
        "--speaker_wav",
        nargs="+",
        help="wav file(s) to condition a multi-speaker TTS model with a Speaker Encoder. You can give multiple file paths. The d_vectors is computed as their average.",
        default=None,
    )       


    if 'linux' in platform:
        parser.add_argument("--default_microphone", default='pulse',
                            help="Default microphone name for SpeechRecognition. "
                                 "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()
    
    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.pause_threshold = 2
    recorder.energy_threshold = args.energy_threshold
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
        
    source = sr.Microphone(sample_rate=16000)
    # Load / Download model
    model = args.model
    if args.model != "large" and not args.non_english:
        model = model + ".en"
    #audio_model = whisper.load_model(model)
    audio_model = WhisperModel(model, device=args.device)
    #audio_model = WhisperModel(model, device="cuda")

    print("Initializing TTS model")
    #if(args.tts_model_path):
    synthesizer = init_tts(args)
    #else:
    #    synthesizer = None

    record_timeout = args.record_timeout
    phrase_timeout = args.phrase_timeout

    temp_file = NamedTemporaryFile().name
    transcription = ['']
    
    with source:
        print("Calibrating mic noise, stay silent...")
        recorder.adjust_for_ambient_noise(source)
        print("OK!")

    def record_callback(_, audio:sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    #recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    listen_stop = recorder.listen_in_background(source, record_callback)

    # Cue the user that we're ready to go.
    cls()
    print(f"{datetime.now()}: Model loaded. OK")
    if args.output_file:
        print(f"Output file: {args.output_file}")


    while True:
        try:
            # Infinite loops are bad for processors, must sleep.
            # Pull raw recorded audio from the queue.
            if data_queue.empty():
                sleep(0.1)
                continue

            now = datetime.utcnow()
            phrase_complete = False
            # If enough time has passed between recordings, consider the phrase complete.
            # Clear the current working audio buffer to start over with the new data.
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                last_sample = bytes()
                phrase_complete = True
            # This is the last time we received new audio data from the queue.
            phrase_time = now
            # Concatenate our current audio data with the latest audio data.
            #while not data_queue.empty():
            #    data = data_queue.get()
            #    last_sample += data

            last_sample = data_queue.get()
            # Use AudioData to convert the raw data to wav data.
            audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
            wav_data = io.BytesIO(audio_data.get_wav_data())
            # Write wav data to the temporary file as bytes.
            with open(temp_file, 'w+b') as f:
                f.write(wav_data.read())
            # Read the transcription.
            #result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
            #segments, info = audio_model.transcribe(temp_file, beam_size=5)
            tstart = time.time()
            print(" *** TRANSCRIBIENDO VOZ")
            segments, info = audio_model.transcribe(temp_file, beam_size=5, language=args.language)
            #segments, info = audio_model.transcribe(temp_file, beam_size=5, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
            tend = time.time()
            ttrans = tend - tstart
            #segments, info = audio_model.transcribe(temp_file, beam_size=1, vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))
            text = ""
            for segment in segments:
                text += segment.text.strip() + ' '
            # If we detected a pause between recordings, add a new item to our transcripion.
            # Otherwise edit the existing one.
            #if phrase_complete:
            #    transcription.append(text)
            #else:
            #    ##transcription[-1] = text
            #    transcription[-1] = text
            text=text.strip()
            if text:
                print(f" *** {datetime.now()}: phrase_complete {phrase_complete} lines {len(transcription)} infer_time {ttrans:.3f}s")
                print(f" *** Text: {text}")
                text_final = convert_num_to_words(text, args.language)
                print(f" *** Text without numbers: {text_final}")
                #transcription.append(text)
                if synthesizer:
                    print(" *** SINTETIZANDO VOZ")
                    listen_stop()
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        #wav = synthesizer.tts(text)
                        tmpfilename = os.path.join(tmpdirname, "temp.wav")

                        # Generating wav...
                        
                        synthesizer.tts_to_file(text_final, speaker_wav=args.speaker_wav, language=args.language, file_path=tmpfilename, split_sentences=False)
                        #synthesizer.tts_with_vc_to_file(text_final, speaker_wav=args.speaker_wav, language=args.language, file_path=tmpfilename, split_sentences=False)

                        # Playing sound...
                        #synthesizer.save_wav(wav, tmpfilename, pipe_out=False)
                        if winsound:
                            winsound.PlaySound(tmpfilename, winsound.SND_FILENAME)
                        else:
                            playsound(tmpfilename)
                    listen_stop = recorder.listen_in_background(source, record_callback)
                    print(" *** ESCUCHANDO")
                    

            # Clear the console to reprint the updated transcription.
        except KeyboardInterrupt:
            break

    print(f" *** {datetime.now()}: transcription end")


if __name__ == "__main__":
    main()

