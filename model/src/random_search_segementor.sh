#!bin/bash

for seed in 1 2 3 4 5 6;
do
    for hidden_size in 16 32 64;
    do
         #model_dir=../models/segmentor/segmentor_12_5_${batch_size}_${seed}/
         log_dir=../outputs/log/segmentor_12_5_${batch_size}_${seed}/
         #mkdir -p ${model_dir}
         mkdir -p ${log_dir}
         python ../src/main.py  \
	 --seed ${seed} \
         --hidden_size ${hidden_size} \
         #--batch_size ${batch_size} \
         &> log_dir/log.txt
    done
done
