rm -f exp.txt
echo '=========================inc-exp-r============================'
echo '=========================inc-exp-r============================' >> exp.txt
python demo/tacas2023/exp11/inc-expr.py vril >> exp.txt
# diff demo/tacas2023/exp1/output_par.json demo/tacas2023/exp1/output_ser.json >> diff.txt
echo '=========================inc-exp-n============================'
echo '=========================inc-exp-n============================' >> exp.txt
python demo/tacas2023/exp11/inc-expr.py vnil >> exp.txt
# diff demo/tacas2023/exp3/output3_par.json demo/tacas2023/exp3/output3_ser.json >> diff.txt
echo '=========================inc-exp-3============================'
echo '=========================inc-exp-3============================' >> exp.txt
python demo/tacas2023/exp11/inc-expr.py v3il >> exp.txt
# diff demo/tacas2023/exp5/output5_par.json demo/tacas2023/exp5/output5_ser.json >> diff.txt
echo '=========================inc-exp-8============================'
echo '=========================inc-exp-8============================' >> exp.txt
python demo/tacas2023/exp11/inc-expr.py v8il >> exp.txt
# diff demo/tacas2023/exp9/output9_dryvr_par.json demo/tacas2023/exp9/output9_dryvr_ser.json >> diff.txt
echo '=========================done============================'