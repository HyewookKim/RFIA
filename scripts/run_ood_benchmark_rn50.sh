#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python rfia_runner.py    --config configs \
                                                --wandb-log \
                                                --datasets I/A/V/R/S \
                                                --backbone RN50 \