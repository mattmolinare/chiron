#!/bin/bash
tmpfile=$(mktemp)
wget https://figshare.com/ndownloader/articles/1512427/versions/5 -O $tmpfile
outdir=mat
unzip $tmpfile -d $outdir
rm $tmpfile
for file in $outdir/*.zip; do
  unzip $file -d ${file%%.zip}
  rm $file
done
