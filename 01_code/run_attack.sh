#!/bin/bash
#
# run_attack.sh is a script which executes the attack



input_folder="../attack_img/attack_img_50_32_6m_baseline" 

arr=(5 7 9)
for (( i=0; i <${#arr[@]}; i++)) ;do
	file=$((${i} + 500))
  	output_folder="../attack_img/attack_img_250_32_6m_multi_250_${file}" 
    
	#判断文件是否存在，不存在则创建该文件
	if [ ! -f $output_folder ]; then
	      mkdir -p ${output_folder}
	fi

   val=${arr[i]}
   #momentum_num=`echo  "0.1 * ${val} "| bc`
  
	echo $output_folder	

  	CUDA_VISIBLE_DEVICES=3 python3 attack_img_50_32_6m_2.py \
	   --output_dir="${output_folder}" \
	   --input_dir="${input_folder}" \
	   --iterations=250 \
	   --batch_size=30\
	   --image_resize=800\
	   --prob=1\
	   --sig=4\
  	   --momentum=1.2\
	   --augment_stddev=0.0066\
	   --rotate_stddev=0.004\
       --kernlen=${val} \
       --max_epsilon=32
	   
done


