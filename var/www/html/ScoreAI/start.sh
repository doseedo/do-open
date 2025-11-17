#!/bin/bash

# Start Uvicorn in the background
uvicorn main:app --host 0.0.0.0 --port 8080
