#!/bin/bash

SCPT_DIR=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)

cd "$SCPT_DIR"

python Code/estimation.py
