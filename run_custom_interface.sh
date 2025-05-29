#!/bin/bash

# Run MelodyFlow custom interface with fine-tuned model support
cd "$(dirname "$0")"
python demos/melodyflow_app_custom.py --inbrowser "$@" 