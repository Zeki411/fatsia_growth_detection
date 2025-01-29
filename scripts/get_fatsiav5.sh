#!/bin/bash

mkdir -p "./data/fatsiav5"

data_dir="./data/fatsiav5"
url="https://app.roboflow.com/ds/qHXWvNjed6?key=GoTLzbFc8M"
f="fatsiav5.zip"

echo " downloading $url to $data_dir/$f ..."

curl -L $url > $data_dir/$f
unzip -q $data_dir/$f -d $data_dir
rm $data_dir/$f
