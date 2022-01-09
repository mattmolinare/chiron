#!/bin/bash
tmpfile=$(mktemp)
wget https://figshare.com/ndownloader/articles/1512427/versions/5 -O $tmpfile
unzip $tmpfile
rm $tmpfile
for file in *.zip; do
  unzip $file -d ${file%%.zip}
  rm $file
done
