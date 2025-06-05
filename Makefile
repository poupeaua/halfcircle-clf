#!/bin/bash

image_name=aitt-symbol-clf
container_name=aitt-symbol-clf-ctr

run:
	fastapi dev api.py

build:
	docker build -t ${image_name} .

run-docker:
	docker run -p 80:8000 ${image_name} --name ${container_name} --rm