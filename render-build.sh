#!/usr/bin/env bash
set -e

apt-get update
apt-get install -y ffmpeg portaudio19-dev

pip install --upgrade pip
pip install -r requirements.txt
