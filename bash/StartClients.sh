#!/bin/bash

start_client() {
	local client_id=$1
	echo "Starting client $client_id"
	python3 client.py $client_id &
}

for i in {1..1}; do
	start_client $i
done

echo "All clients have started"
