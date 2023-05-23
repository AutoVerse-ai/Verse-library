#/bin/bash

if [[ $1 == *l* ]]
then
    para='l'
else 
    para=''
fi
echo $para

file="expr_rst.txt"
rm -f $file

# echo '=========================exp1============================' >> $file
python demo/tacas2023/exp1/exp1.py "${para}" >> $file
echo '=========================exp9_dryvr============================' >> $file
python demo/tacas2023/exp9/exp9_dryvr.py "${para}" >> $file
echo '=========================exp10_dryvr============================' >> $file
python demo/tacas2023/exp10/exp10_dryvr.py "${para}" >> $file
echo '=========================exp3============================' >> $file
python demo/tacas2023/exp3/exp3.py "${para}" >> $file
echo '=========================exp2_straight============================' >> $file
python demo/tacas2023/exp2/exp2_straight.py "${para}" >> $file
echo '=========================exp2_curve============================' >> $file
python demo/tacas2023/exp2/exp2_curve.py "${para}" >> $file
echo '=========================exp5============================' >> $file
python demo/tacas2023/exp5/exp5.py "${para}" >> $file
echo '=========================exp4_noise============================' >> $file
python demo/tacas2023/exp4/exp4_noise.py "${para}" >> $file
echo '=========================exp6_dryvr============================' >> $file
python demo/tacas2023/exp6/exp6_dryvr.py "${para}" >> $file
echo '=========================vanderpol_demo2============================' >> $file
python demo/tacas2023/exp12/vanderpol_demo2.py "${para}" >> $file
echo '=========================rendezvous_demo============================' >> $file
python demo/tacas2023/exp12/rendezvous_demo.py "${para}" >> $file
echo '=========================gearbox_demo============================' >> $file
python demo/tacas2023/exp12/gearbox_demo.py "${para}" >> $file