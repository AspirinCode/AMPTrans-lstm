#!/bin/bash

#folder="/home/shun/LSTM_peptides/pdb_train_lstm"
folder="/root/autodl-tmp/LSTM_peptides/pdb_train_lstm"

echo "run LSTM_peptides fine_tune..."

softfiles=$(ls $folder)

last_file="pdb"

for sfile in ${softfiles} 
do  
	echo "process:" $folder/${sfile}	
	

	python LSTM_peptides.py --name ${sfile%.*}  --dataset $folder/${sfile} --modfile ./$last_file/checkpoint/model_epoch_22.hdf5 --epochs 30  
	
	echo "finish:" $folder/${sfile}
	
	last_file=${sfile%.*}
	

done
