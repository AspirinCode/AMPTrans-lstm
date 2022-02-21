#!/bin/bash

folder="./train_data/uniprot_train_transformer"

echo "run TRANS_peptides 1..."

softfiles=$(ls $folder)

#model_file='model.pt'

model_file='final_finetuning_model.pt'

for sfile in ${softfiles} 
do  
	echo "process:" $folder/${sfile}
	
	python finetuning.py 1 $model_file ${sfile}
	
	echo "finish:" $folder/${sfile}
	
	
	

done

