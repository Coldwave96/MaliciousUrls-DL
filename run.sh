emb_modes=(1 2 3 4 5)
nb_epoch=5

for ((i=0; i <${#emb_modes[@]}; ++i))
    do
    python train.py --data.malicious_data IntegratedData/malicious_train.txt --data.benign_data IntegratedData/benign_train.txt \
    --model.emb_mode ${emb_modes[$i]} --train.nb_epochs ${nb_epoch} --train.batch_size 1048 \
    --log.output_dir Model/runs/emb${emb_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/

    python test.py --data.data_dir  IntegratedData/test.csv\
    --data.word_dict_dir Model/runs/emb${emb_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/words_dict.pickle \
    --data.subword_dict_dir Model/runs/emb${emb_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/subwords_dict.pickle \
    --data.char_dict_dir Model/runs/emb${emb_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/chars_dict.pickle \
    --log.checkpoint_dir Model/runs/emb${emb_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/checkpoints/ \
    --log.output_dir Model/runs/emb${emb_modes[$i]}_32dim_minwf1_1conv3456_${nb_epoch}ep/train_test.txt \
    --model.emb_mode ${emb_modes[$i]} --model.emb_dim 32 --test.batch_size 1048
    done
