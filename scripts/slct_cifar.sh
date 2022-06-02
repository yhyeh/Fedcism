LR=0.1
LE=1
EP=1000
DATA=cifar10
NCLS=10
DIR=cossim_htail_vi

for VI in 3 1 5; do
  for ALGO in 0; do
    for FRAC in 0.1 0.05; do
      
      
      # shard 10 fed
      python3 main_fed.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
        --num_users 100 --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save ${DIR}${VI} \
        --cls_imb --vol_imb ${VI} --clsimb_type htail \
        --data_distr htail_dict_users_vi${VI}.pkl
      
      # shard 2 fed
      python3 main_fed.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
        --num_users 100 --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save ${DIR}${VI} \
        --cls_imb  --vol_imb ${VI} --clsimb_type htail \
        --data_distr htail_dict_users_vi${VI}.pkl
      
      :'
      for E in 0.8; do

      
        # shard 10 oort
        python3 main_slct_algo3.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
          --num_users 100 --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
          --results_save ${DIR} --myalgo ${ALGO} \
          --cls_imb  --clsimb_type zipf \
          --data_distr zipf_dict_users.pkl \
          --epsilon ${E}
        
        
        # shard 2 oort
        python3 main_slct_algo3.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
          --num_users 100 --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
          --results_save ${DIR}  --myalgo ${ALGO} \
          --cls_imb  --clsimb_type zipf \
          --data_distr zipf_dict_users.pkl \
          --epsilon ${E}
        
      
      
      done
      '
    done
  done
done