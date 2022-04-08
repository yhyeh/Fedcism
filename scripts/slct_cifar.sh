LR=0.1
LE=1
EP=1000
for RUN in cossim_zipf; do
  for ALGO in 0; do
    for FRAC in 0.1 0.05; do
      
      :'
      # shard 10 fed
      python3 main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
        --num_users 100 --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save ${RUN} \
        --cls_imb  --clsimb_type zipf \
        --data_distr zipf_dict_users.pkl  
      
      # shard 2 fed
      python3 main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
        --num_users 100 --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save ${RUN} \
        --cls_imb  --clsimb_type zipf \
        --data_distr zipf_dict_users.pkl  
      '

      for E in 0.5 0.8; do

      
        # shard 10 oort
        python3 main_slct_algo3.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
          --num_users 100 --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
          --results_save ${RUN} --myalgo ${ALGO} \
          --cls_imb  --clsimb_type zipf \
          --data_distr zipf_dict_users.pkl \
          --epsilon ${E}   
        
        :'
        # shard 2 oort
        python3 main_slct_algo3.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
          --num_users 100 --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
          --results_save ${RUN}  --myalgo ${ALGO} \
          --cls_imb  --clsimb_type zipf \
          --data_distr zipf_dict_users.pkl \
          --epsilon ${E}    
        '
      
      
      done
    done
  done
done