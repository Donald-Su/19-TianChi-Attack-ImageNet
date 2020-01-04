#!/bin/sh

ORG_FOLDER="images"
ATTCK_FOLDER=$1
Cal_attack=$2
Csv_file=$3


# 注意：如果使用攻击的图片进行准确率识别，分数低是正常现象

# 使用inception_v3
CUDA_VISIBLE_DEVICES=0 python3 ./05_acc/tf_score_prediction.py \
    --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/inception_v3.ckpt"\
    --scope_name="InceptionV3"\
    --img_folder="${ATTCK_FOLDER}"\
    --cal_attack="${Cal_attack}"\
    --csv_file="${Csv_file}"

# 使用adv_inception_v3
CUDA_VISIBLE_DEVICES=0 python3 ./05_acc/tf_score_prediction.py \
    --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/adv_inception_v3.ckpt"\
    --scope_name="InceptionV3"\
    --img_folder="${ATTCK_FOLDER}"\
    --cal_attack="${Cal_attack}"\
    --csv_file="${Csv_file}"

# # 使用ens3_adv_inception_v3
# python3 ./05_acc/tf_score_prediction.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/ens3_adv_inception_v3.ckpt"\
#     --scope_name="InceptionV3"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"
    
# # 使用ens4_adv_inception_v3
# python3 ./05_acc/tf_score_prediction.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/ens4_adv_inception_v3.ckpt"\
#     --scope_name="InceptionV3"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"
    

# # 使用adv_inception_resnet_v2
# python3 ./05_acc/tf_score_prediction_v2.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/adv_inception_resnet_v2.ckpt"\
#     --scope_name="InceptionResnetV2"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"
    
    
# # 使用ens_adv_inception_resnet_v2
# python3 ./05_acc/tf_score_prediction_v2.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/ens_adv_inception_resnet_v2.ckpt"\
#     --scope_name="InceptionResnetV2"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"


#  # 使用resnet_v2_152
# python3 ./05_acc/tf_score_prediction_resnet.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/resnet_v2_152.ckpt"\
#     --scope_name="resnet_v2_152"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"


# #  # 使用resnet_v2_101
# python3 ./05_acc/tf_score_prediction_resnet.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/resnet_v2_101.ckpt"\
#     --scope_name="resnet_v2_101"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"

# #  使用resnet_v2_50
# python3 ./05_acc/tf_score_prediction_resnet.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/resnet_v2_50.ckpt"\
#     --scope_name="resnet_v2_50"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"

#  # 使用inception_v4
# python3 ./05_acc/tf_score_prediction_v4.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/inception_v4.ckpt"\
#     --scope_name="InceptionV4"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"

#  # 使用inception_v4
# python3 ./05_acc/tf_score_prediction_v4.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/inception_v4.ckpt"\
#     --scope_name="InceptionV4"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"


 # 使用vgg19
# python3 ./05_acc/tf_vgg19.py \
#     --ckpt_file="/home/suy/data_disk/datasets_sdc/01_model_weight/vgg_19.ckpt"\
#     --scope_name="vgg_19"\
#     --img_folder="${ATTCK_FOLDER}"\
#     --cal_attack="${Cal_attack}"\
#     --csv_file="${Csv_file}"

