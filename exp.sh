rm -f exp.txt
echo '=========================exp1============================'
echo '=========================exp1============================' >> exp.txt
python demo/tacas2023/exp1/exp1.py  >> exp.txt
diff demo/tacas2023/exp1/output_par.json demo/tacas2023/exp1/output_ser.json >> diff.txt
echo '=========================exp3============================'
echo '=========================exp3============================' >> exp.txt
python demo/tacas2023/exp3/exp3.py >> exp.txt
diff demo/tacas2023/exp3/output3_par.json demo/tacas2023/exp3/output3_ser.json >> diff.txt
echo '=========================exp5============================'
echo '=========================exp5============================' >> exp.txt
python demo/tacas2023/exp5/exp5.py >> exp.txt
diff demo/tacas2023/exp5/output5_par.json demo/tacas2023/exp5/output5_ser.json >> diff.txt
echo '=========================exp9============================'
echo '=========================exp9============================' >> exp.txt
python demo/tacas2023/exp9/exp9_dryvr.py >> exp.txt
iff demo/tacas2023/exp9/output9_dryvr_par.json demo/tacas2023/exp9/output9_dryvr_ser.json >> diff.txt
echo '=========================exp11============================'
echo '=========================exp11============================' >> exp.txt
python demo/tacas2023/exp11/inc-expr.py vbd >> exp.txt
diff demo/tacas2023/exp11/main_par.json demo/tacas2023/exp11/main_ser.json >> diff.txt
echo '=========================done============================'