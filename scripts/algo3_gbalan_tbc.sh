LR=0.1
LE=1
EP=500
DATA=cifar10
NCLS=10
E=0.8
DEG=1
DIR=tbc_gbalan_vi

#for RUN in cls_imb; do
#for RUN in cossim_test; do
for RUN in 0; do

  for FRAC in 0.05 0.1; do
    for VI in 3; do
      
      # shard 10 fed
      python3 main_fed_tbc.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
        --num_users 100 --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save ${DIR}${VI}_r${RUN} \
        --vol_imb ${VI} \
        --data_distr gbalan_dict_users_vi${VI}_r${RUN}.pkl
        
      # shard 2 fed
      python3 main_fed_tbc.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
        --num_users 100 --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save ${DIR}${VI}_r${RUN} \
        --vol_imb ${VI} \
        --data_distr gbalan_dict_users_vi${VI}_r${RUN}.pkl
      
      for ALGO in 3 0; do
      
        
        # shard 10 fed w/ sel
        python3 main_slct_tbc.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
          --num_users 100 --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
          --results_save ${DIR}${VI}_r${RUN} --myalgo ${ALGO} --deg ${DEG} --wndw_size 100 \
          --vol_imb ${VI} --data_distr gbalan_dict_users_vi${VI}_r${RUN}.pkl \
          --epsilon ${E} \
          --gpu 2
        
        # shard 2 fed w/ sel
        python3 main_slct_tbc.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
          --num_users 100 --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
          --results_save ${DIR}${VI}_r${RUN} --myalgo ${ALGO} --deg ${DEG} --wndw_size 100 \
          --vol_imb ${VI} --data_distr gbalan_dict_users_vi${VI}_r${RUN}.pkl \
          --epsilon ${E} \
          --gpu 2
      
      done
    done
  done
done
