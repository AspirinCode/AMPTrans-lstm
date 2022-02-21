#!/bin/bash

folder="./train_data/uniprot_train_transformer"

echo "run TRANS_peptides 1..."

softfiles=$(ls $folder)

model_file='model.pt'

for sfile in ${softfiles} 
do  
	echo "process:" $folder/${sfile}
	
	python finetuning.py 1 $model_file ${sfile}
	
	echo "finish:" $folder/${sfile}
	
	model_file='final_finetuning_model.pt'
	

done

folder="./pretrain_data/pdb_train_transformer"

echo "run TRANS_peptides 2..."

softfiles=$(ls $folder)

model_file='final_finetuning_model.pt'

for sfile in ${softfiles} 
do  
	echo "process:" $folder/${sfile}
	
	python finetuning.py 2 $model_file ${sfile}
	
	echo "finish:" $folder/${sfile}

done