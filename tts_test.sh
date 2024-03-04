#!/bin/sh

BASE_DIR_MODEL=model_tts/vits_jacalles.20240301

echo Texto: $*

tts \
  --text "$*" \
  --model_path ${BASE_DIR_MODEL}/best_model.pth \
  --config_path ${BASE_DIR_MODEL}/config.json \
  --use_cuda false --pipe | aplay
