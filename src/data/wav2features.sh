#!/usr/bin/env bash

echo "Hello Wrokd!"

indir="../../data/raw/wav/"
conffile="/home/christian/bin/opensmile-2.3.0/config/IS13_ComParE.conf"
outfile="../../data/processed/data.csv"

echo $indir
echo $conffile
echo $outfile

# TODO: using env variables from .env or so
# SMILExtract="~/bin/opensmile-2.3.0/bin/linux_x64_standalone_static/SMILExtract"
# rm -fv ${outdir}"*.csv"

for fname in ${indir}*.wav; do
    if [ -f ${fname} ]; then
        echo "processing "${fname}
        SMILExtract -C ${conffile} -I ${fname} -start 0 -end -1  -csvoutput ${outfile} -appendcsv 1
    else
        echo "File not found!"
    fi
done
