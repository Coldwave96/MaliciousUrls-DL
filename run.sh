emb_modes=(1 2 3 4 5)
nb_epoch=5

for ((i=0; i <${#emb_modes[@]}; ++i))
    do
    python train.py --data.malicious_data IntegratedData/malicious.txt --data.benign_data IntegratedData/benign.txt \
    --model.emb_mode ${emb_modes[$i]} --train.nb_epochs ${nb_epoch} --train.batch_size 1048 \
    --log.output_dir Model/runs_1/emb${emb_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/
    done
