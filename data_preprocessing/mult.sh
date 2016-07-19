#!/bin/bash
thr=+0

parallel --bar -j $thr --header : "/Applications/Slicer\ 5.app/Contents/MacOS/Slicer" --no-splash --python-script ~/Dropbox/MachineLearning/script.py -s '1,1,1' -k {i} ::: i `seq 0 1 70`
 
