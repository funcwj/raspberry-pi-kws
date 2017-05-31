#!/bin/bash

arecord -f S16_LE -r 16000 -d 10 | ./query_by_pipe.py || exit 1
