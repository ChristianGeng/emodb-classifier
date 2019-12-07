#!/usr/bin/env bash

echo "Hello Wrokd!"
indir="../../data/raw/wav/"
conffile="/home/christian/bin/opensmile-2.3.0/config/IS13_ComParE.conf"
outfile_features="../../data/processed/features.csv"
outfile_labels="../../data/processed/filenames.csv"

echo $indir
echo $conffile
echo $outfile_features

# TODO: using env variables from .env or so
# SMILExtract="~/bin/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract"
rm -f ${outfile_features}
echo "filename" > ${outfile_labels}

for fname in ${indir}*.wav; do
    if [ -f ${fname} ]; then
        echo "processing "${fname}
        SMILExtract -C ${conffile} -I ${fname} -start 0 -end -1  -csvoutput ${outfile_features} -appendcsv 1
        echo $fname
        filename=$(basename -- "$fname")
        extension="${filename##*.}"
        filename="${filename%.*}"
        echo ${filename}
        # emotion=$(echo ${filename} | cut  -c 6)
        echo ${filename} >> ${outfile_labels}
    else
        echo "File not found!"
    fi
done
