#/bin/bash

para=''
inc='i'

file="exp_v${inc}${para}.txt"
rm -f $file

echo '=========================inc-exp-r============================'
echo '=========================inc-exp-r============================' >> $file
python demo/tacas2023/exp11/inc-expr.py "vr${inc}${para}" >> $file
# diff demo/tacas2023/exp1/output_par.json demo/tacas2023/exp1/output_ser.json >> diff.txt
echo '=========================inc-exp-n============================'
echo '=========================inc-exp-n============================' >> $file
python demo/tacas2023/exp11/inc-expr.py "vn${inc}${para}" >> $file
# diff demo/tacas2023/exp3/output3_par.json demo/tacas2023/exp3/output3_ser.json >> diff.txt
echo '=========================inc-exp-8============================'
echo '=========================inc-exp-8============================' >> $file
python demo/tacas2023/exp11/inc-expr.py "v8${inc}${para}" >> $file
# diff demo/tacas2023/exp9/output9_dryvr_par.json demo/tacas2023/exp9/output9_dryvr_ser.json >> diff.txt
echo '=========================done============================'