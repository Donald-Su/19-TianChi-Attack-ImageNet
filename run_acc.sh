#!/bin/bash

# Evaluate the original image score and the effect of the attack

# Cal_attack: 
#	False: Evaluate the original image score
#	True:  Evaluate the effect of the attack

# for i in $(seq 20 20 140);do

        ATTCK_FOLDER="./attack_img/attack_img_250_32_6m_multi_250_500"
        #ATTCK_FOLDER="images"
        Cal_attack="True"
        Csv_file='dev.csv'

        ./acc_attack.sh ${ATTCK_FOLDER}  ${Cal_attack}  ${Csv_file}

# done

