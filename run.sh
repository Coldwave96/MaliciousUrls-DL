# emb_modes=(1 2 2 3 3 4 5)
# delimit_modes=(0 0 1 0 1 1 1)
emb_modes=(1)
delimit_modes=(0)
nb_epoch=5

for ((i=0; i <${#emb_modes[@]}; ++i))
    do
    python train.py --data.malicious_data IntegratedDataDemo/malicious.txt --data.benign_data IntegratedDataDemo/benign.txt \
    --data.delimit_mode ${delimit_modes[$i]} --model.emb_mode ${emb_modes[$i]} --train.nb_epochs ${nb_epoch} \
    --train.batch_size 1048 --log.output_dir Model/runs_1/emb${emb_modes[$i]}_dlm${delimit_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/
    done
