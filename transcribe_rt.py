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
import yaml
from rvc import Config, load_hubert, get_vc, rvc_infer
import gc
import torch


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


def init_tts(config):
        print("Initializing TTS model")
        # Late-import to make things load faster
        from TTS.api import TTS
        from TTS.utils.manage import ModelManager
        from TTS.utils.synthesizer import Synthesizer

        tts = TTS(model_path=config['xtts_model_path'], config_path=config['xtts_model_config'], progress_bar=True, gpu=True).to(config['device'])

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





def init_rvc(config):
    print("Initializing RVC model")
    rvc=config['rvc_model_pth']
    modelname = os.path.splitext(rvc)[0]
    print("Using RVC model: "+ modelname)
    rvc_model_path = config['rvc_model_pth']
    rvc_index_path = config['rvc_model_index']

    if rvc_index_path != "" :
        print("Index file found!")

    #load_cpt(modelname, rvc_model_path)
    #cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    is_half = device != 'cpu'

    class RVC_Data: 
        def __init__(self):
            self.current_model = {}
            self.cpt = {}
            self.version = {}
            self.net_g = {}
            self.tgt_sr = {}
            self.vc = {} 

        def load_cpt(self, modelname, rvc_model_path):
            if self.current_model != modelname:
                    print("Loading new model")
                    del self.cpt, self.version, self.net_g, self.tgt_sr, self.vc
                    #device = config['device']
                    config_class = Config(device, is_half)
                    self.cpt, self.version, self.net_g, self.tgt_sr, self.vc = get_vc(device, is_half, config_class, rvc_model_path)
                    self.current_model = modelname 

    rvc_data = RVC_Data()
    rvc_data.load_cpt(modelname, rvc_model_path)
    hubert_model = load_hubert(device, is_half, "./models/hubert_base.pt")

    return rvc_data, hubert_model


def voice_change(rvc_data, pitch_change, index_rate):
    rvc_infer(
        index_path=rvc_index_path, 
        index_rate=index_rate, 
        input_path="./output.wav", 
        output_path="./outputrvc.wav", 
        pitch_change=pitch_change, 
        f0_method="rmvpe", 
        cpt=rvc_data.cpt, 
        version=rvc_data.version, 
        net_g=rvc_data.net_g, 
        filter_radius=3, 
        tgt_sr=rvc_data.tgt_sr, 
        rms_mix_rate=0.25, 
        protect=0, 
        crepe_hop_length=0, 
        vc=rvc_data.vc, 
        hubert_model=hubert_model
    )
    gc.collect()

def sendsound_out(filepath):
    # Playing sound...
    #synthesizer.save_wav(wav, tmpfilename, pipe_out=False)
    if winsound:
        winsound.PlaySound(filepath, winsound.SND_FILENAME)
    else:
        playsound(filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yml", help="Config file")
    parser.add_argument("--rvc", action='store_true', default=False, help="Use RVC")

    #parser.add_argument("--model", default="large-v2", help="Model to use",
    #                    choices=["tiny", "base", "small", "medium", "large", "large-v1", "large-v2", "large-v3"])
    #parser.add_argument("--non_english", action='store_true', default=True,
    #                    help="Don't use the english model.")
    #parser.add_argument("--energy_threshold", default=1000,
    #                    help="Energy level for mic to detect.", type=int)
    #parser.add_argument("--record_timeout", default=2,
    #                    help="How real time the recording is in seconds.", type=float)
    #parser.add_argument("--phrase_timeout", default=3,
    #                    help="How much empty space between recordings before we "
    #                         "consider it a new line in the transcription.", type=float)  
    #parser.add_argument("--tts_model_name", default=None,
    #                    help="TTS model name", type=str)
    #parser.add_argument("--tts_model_path", default=None,
    #                    help="Path to TTS model", type=str)
    #parser.add_argument("--tts_model_config", default=None,
    #                    help="Path to TTS config", type=str)
    #parser.add_argument("--language", default=None,
    #                    help="Language", type=str)
#
    #parser.add_argument("--output_file", default=None,
    #                    help="Output text to file.", type=str)
    #parser.add_argument("--device", default="cuda",
    #                    help="CUDA or cpu", type=str)
    #parser.add_argument(
    #    "--speaker_wav",
    #    nargs="+",
    #    help="wav file(s) to condition a multi-speaker TTS model with a Speaker Encoder. You can give multiple file paths. The d_vectors is computed as their average.",
    #    default=None,
    #)       

    #if 'linux' in platform:
    #    parser.add_argument("--default_microphone", default='pulse',
    #                        help="Default microphone name for SpeechRecognition. "
    #                             "Run this with 'list' to view available Microphones.", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # The last time a recording was retreived from the queue.
    phrase_time = None
    # Current raw audio bytes.
    last_sample = bytes()
    # Thread safe Queue for passing data from the threaded recording callback.
    data_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.pause_threshold = config['pause_threshold']
    recorder.energy_threshold = config['energy_threshold']
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False
        
    source = sr.Microphone(sample_rate=16000)
    # Load / Download model
    model = config['whisper_model']
    #if not model.startswith("large") and not args.non_english:
    #    model = model + ".en"
    #audio_model = whisper.load_model(model)
    audio_model = WhisperModel(model, device=config['device'])
    #audio_model = WhisperModel(model, device="cuda")

    
    #if(args.tts_model_path):
    synthesizer = init_tts(config)
    if args.rvc and config['rvc_model_pth']:
        rvc_model, hubert_model = init_rvc(config)
    else:
        rvc_model, hubert_model = None, None

    #else:
    #    synthesizer = None

    record_timeout = config['record_timeout']
    phrase_timeout = config['phrase_timeout']

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
    if config['output_file']:
        print(f"Output file: {config['output_file']}")

    print(f"{datetime.now()}: PROCESANDO: RVC: {'ENABLED' if rvc_model else 'DISABLED'}")

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
            segments, info = audio_model.transcribe(temp_file, beam_size=5, language=config['language'])
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
                text_final = convert_num_to_words(text, config['language'])
                print(f" *** Text without numbers: {text_final}")
                #transcription.append(text)
                if synthesizer:
                    print(f"{datetime.now()}: PARANDO ESCUCHA")
                    listen_stop()
                    with tempfile.TemporaryDirectory() as tmpdirname_synth:
                        #wav = synthesizer.tts(text)
                        tmpfilename_synth = os.path.join(tmpdirname_synth, "synth.wav")

                        # Generating wav...
                        print(f"{datetime.now()}: SINTETIZANDO VOZ")    
                        synthesizer.tts_to_file(text_final, 
                            speaker_wav=config['xtts_speaker_wav'], 
                            language=config['language'], 
                            file_path=tmpfilename_synth,
                            split_sentences=False,
                            temperature=config['xtts_model_temperature'],
                            length_penalty=config['xtts_length_penalty'],
                            repetition_penalty=config['xtts_repetition_penalty'],
                            top_k=config['xtts_top_k'],
                            top_p=config['xtts_top_p'],
                            num_beams=config['xtts_num_beams'],
                            speed=config['xtts_speed'])

                        if rvc_model:
                            print(f"{datetime.now()}: CAMBIANDO TIMBRE DE VOZ")
                            with tempfile.TemporaryDirectory() as tmpdirname_rvc:
                                #wav = synthesizer.tts(text)
                                tmpfilename_rvc = os.path.join(tmpdirname_rvc, "rvc.wav")

                                # Generating wav...
                                rvc_infer(
                                    index_path=config['rvc_model_pth'],
                                    index_rate=config['rvc_index_rate'],
                                    input_path=tmpfilename_synth,
                                    output_path=tmpfilename_rvc,
                                    pitch_change=config['rvc_pitch_change'],
                                    f0_method="rmvpe",
                                    cpt=rvc_model.cpt,
                                    version=rvc_model.version,
                                    net_g=rvc_model.net_g,
                                    filter_radius=3,
                                    tgt_sr=rvc_model.tgt_sr,
                                    rms_mix_rate=0.25,
                                    protect=0,
                                    crepe_hop_length=0,
                                    vc=rvc_model.vc,
                                    hubert_model=hubert_model)
                                sendsound_out(tmpfilename_rvc)
                        else:
                            sendsound_out(tmpfilename_synth)

                        print(f"{datetime.now()}: ESCUCHANDO")
                        listen_stop = recorder.listen_in_background(source, record_callback)

            # Clear the console to reprint the updated transcription.
        except KeyboardInterrupt:
            break

    print(f" *** {datetime.now()}: transcription end")


if __name__ == "__main__":
    main()

