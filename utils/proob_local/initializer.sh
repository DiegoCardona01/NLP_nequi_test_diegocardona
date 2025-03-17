#!/bin/bash

gunicorn --bind 0.0.0.0:8000 main:app -w 2 -k uvicorn.workers.UvicornWorker