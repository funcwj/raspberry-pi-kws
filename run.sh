#!/bin/bash

arecord -D plughw:1,0 -r 16000 -f S16_LE -t raw -c 1 -d 60 | ./query_by_raspberry.py | ./blink_service.py || echo "pipe chain exited!" && exit 1
