LR=0.1
LE=1
EP=1000
DATA=cifar10
NCLS=10
E=0.8
DEG=1
DIR=cossim_zipf_vi
VI=3

#for RUN in cossim_test; do

for NU in 200 300 400; do
  for RUN in 0 1 2; do

    for FRAC in 0.05; do
      :'
      # shard 10 fed
      python3 main_fed.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
        --num_users ${NU} --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save ${DIR}${VI}_r${RUN} \
        --cls_imb --vol_imb ${VI} --clsimb_type zipf \
        --data_distr zipf_dict_users_vi${VI}_r${RUN}.pkl
      '
      # shard 2 fed
      python3 main_fed.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
        --num_users ${NU} --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save ${DIR}${VI}_r${RUN} \
        --cls_imb  --vol_imb ${VI} --clsimb_type zipf \
        --data_distr zipf_dict_users_vi${VI}_r${RUN}.pkl
      
      for ALGO in 3 0; do
        :'
        # shard 10 fed w/ sel
        python3 main_slct_algo3.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
          --num_users ${NU} --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
          --results_save ${DIR}${VI}_r${RUN} --myalgo ${ALGO} --deg ${DEG} --wndw_size 100 \
          --cls_imb --vol_imb ${VI} --data_distr zipf_dict_users_vi${VI}_r${RUN}.pkl \
          --epsilon ${E} --clsimb_type zipf \
          --gpu 2
        '
        # shard 2 fed w/ sel
        python3 main_slct_algo3.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
          --num_users ${NU} --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
          --results_save ${DIR}${VI}_r${RUN} --myalgo ${ALGO} --deg ${DEG} --wndw_size 100 \
          --cls_imb --vol_imb ${VI} --data_distr zipf_dict_users_vi${VI}_r${RUN}.pkl \
          --epsilon ${E} --clsimb_type zipf \
          --gpu 2
      
      done
    done
  done
done
