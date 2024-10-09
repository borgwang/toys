#! /bin/bash

echo "===== c ====="
gcc nn.c -Wall -o nn -O3 -Wall && ./nn && rm nn

echo "===== rust ====="
rustc nn.rs -C opt-level=3 && ./nn && rm nn

echo "===== python ====="
python nn.py

echo "===== typescript ====="
npx tsx nn.ts
