#!/bin/bash

if [ -e ml-student-competition.zip ]
then
    echo "Competition's data already downloaded..."
else
    echo "Downloading competitions's data..."
    kaggle competitions download -c ml-student-competition
fi

rm -rf Data/
echo "Decompressing Zip to folder Data/"
unzip ml-student-competition -d Data/
