#! /bin/bash
rustc main.rs -C opt-level=3 && ./main && rm main
