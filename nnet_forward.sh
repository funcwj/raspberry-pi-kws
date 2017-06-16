#!/bin/bash 
[ $# -ne 1 ] && echo "format error: $0 [wave-in]" && exit 1
[ -z $KALDI_ROOT ] && echo "export KALDI_ROOT first!" && exit 1

wave_path=true
[ -d $1 ] && wave_path=false


if $wave_path; then
    wave_base=$(dirname $1)
    wave_name=$(basename $1)
    wave_id=${wave_name%.*}
    echo "$wave_id   $1" > wav.scp
else
    wave_base=$(basename $1)
    for wave in $(ls $1); do
        wave_name=$(basename $wave)
        wave_id=${wave_name%.*}
        echo "$wave_id  $wave_base/$wave"
    done > wav.scp
    wave_id="templ_avg"
fi


compute-fbank-feats --config=fbank.conf scp:wav.scp ark:- | apply-cmvn --norm-vars=true cmvn.global ark:- ark:- | \
    splice-feats --left-context=10 --right-context=5 ark:- ark:- | nnet-forward final.nnet ark:- ark:${wave_base}/${wave_id}.post || exit 1

rm wav.scp
echo "OK"
