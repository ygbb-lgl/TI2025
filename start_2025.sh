#!/bin/bash
echo 'startup'
#sleep 30
sudo chmod 777 /dev/ttyTHS1
sudo chmod 777 /dev/ttyACM0
source /home/jetson/archiconda3/bin/activate ocr
cd /home/jetson/2025
python /home/jetson/2025/main.py
echo 'over'
#sleep 30
