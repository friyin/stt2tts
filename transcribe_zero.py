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
from pydub import AudioSegment

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


def init_tts(device):
        print(f"{datetime.now()}: Inicializando modelo XTTS")
        # Late-import to make things load faster
        from TTS.api import TTS
        from TTS.utils.manage import ModelManager
        from TTS.utils.synthesizer import Synthesizer

        #tts = TTS(model_path=config['xtts_model_path'], config_path=config['xtts_model_config'], progress_bar=True, gpu=True).to(config['device'])
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True).to(device)
        return tts


def sendsound_out(filepath):
    # Playing sound...
    #synthesizer.save_wav(wav, tmpfilename, pipe_out=False)
    if winsound:
        winsound.PlaySound(filepath, winsound.SND_FILENAME)
    else:
        playsound(filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", default="speaker.wav", help="Reference wav", type=str)
    parser.add_argument("--lang", default="es", help="Language", type=str)
    parser.add_argument("--device", default="cuda", help="Inference device", type=str)

    args = parser.parse_args()

    synthesizer = init_tts(args.device)

    print(f"{datetime.now()}: Modelo cargado en memoria correctamente")

    while True:
        text = input("Teclee texto para sintetizar voz: ")
        print("")
        if not text:
            print(" *** No hay texto, salgo.")
            break

        if text:
            print(f" *** Text: {text}")
            text_final = convert_num_to_words(text, args.lang)
            if text_final[-1]!='.':
                text_final = text_final + "."
            print(f" *** Text without numbers: {text_final}")
            #transcription.append(text)
            if synthesizer:
                with tempfile.TemporaryDirectory() as tmpdirname_synth:
                    #wav = synthesizer.tts(text)
                    tmpfilename_synth = os.path.join(tmpdirname_synth, "synth.wav")

                    # Generating wav...
                    print(f"{datetime.now()}: SINTETIZANDO VOZ")    
                    synthesizer.tts_to_file(text_final, 
                        speaker_wav=args.wav, 
                        language=args.lang, 
                        file_path=tmpfilename_synth,
                        num_beams=2,
                        split_sentences=True)

                    sendsound_out(tmpfilename_synth)
        print(f" *** {datetime.now()}: transcription end")


if __name__ == "__main__":
    main()

