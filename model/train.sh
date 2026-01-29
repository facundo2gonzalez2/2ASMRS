#!/bin/bash

source /home/fgbarnator/2ASMRS/.venv/bin/activate
nohup python /home/fgbarnator/2ASMRS/model/run_vae.py > run_vae.log 2>&1 &