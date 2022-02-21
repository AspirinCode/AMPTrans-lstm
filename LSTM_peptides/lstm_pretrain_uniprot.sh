#!/bin/bash

#folder="/home/shun/LSTM_peptides/pdb_train_lstm"
folder="/root/autodl-tmp/LSTM_peptides/uniprot_train_lstm"

echo "run LSTM_peptides fine_tune.2.."

softfiles=$(ls $folder)

last_file="18_pdb_train_lstm"

for sfile in ${softfiles} 
do  
	echo "process:" $folder/${sfile}	
	

	python LSTM_peptides.py --name ${sfile%.*}  --dataset $folder/${sfile} --modfile ./$last_file/checkpoint/model_epoch_29.hdf5 --epochs 30  
	
	echo "finish:" $folder/${sfile}
	
	last_file=${sfile%.*}
	

done
