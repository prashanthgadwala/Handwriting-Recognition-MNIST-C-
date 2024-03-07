#!/bin/bash

# Set directory
cd /Users/prashanthgadwala/Documents/Study\ material/Advance\ PT/ProjectFinalNN/ws2023-group-15-idkcoding

# Compile training and testing scripts
g++ src/train.cpp -o train -Isrc -Ipytorch/include -Lpytorch/lib -ltorch -ltorch_cpu -lc10 -lpthread -D_GLIBCXX_USE_CXX11_ABI=0
g++ src/test.cpp -o test -Isrc -Ipytorch/include -Lpytorch/lib -ltorch -ltorch_cpu -lc10 -lpthread -D_GLIBCXX_USE_CXX11_ABI=0
