LR=0.1
LE=1
EP=1000
DATA=cifar10
NCLS=10
#for RUN in cls_imb; do
for RUN in cossim_zipf1; do

  # main_slct.py , adaptive gamma w.r.t. loss
  for FRAC in 0.05 0.1; do
    for ALGO in 3 0; do
      for DEG in 1; do
        for E in 0.8; do
        
          
          # shard 10 fed w/ sel
          python3 main_slct_unitg.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
            --num_users 100 --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
            --results_save ${RUN} --myalgo ${ALGO} --deg ${DEG} --wndw_size 100 \
            --cls_imb --data_distr zipf_dict_users.pkl \
            --epsilon ${E} --clsimb_type zipf \
            --gpu 2
        
          # shard 2 fed w/ sel
          python3 main_slct_unitg.py --dataset ${DATA} --model cnn --num_classes ${NCLS} --epochs ${EP} --lr ${LR} \
            --num_users 100 --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
            --results_save ${RUN} --myalgo ${ALGO} --deg ${DEG} --wndw_size 100 \
            --cls_imb --data_distr zipf_dict_users.pkl \
            --epsilon ${E} --clsimb_type zipf \
            --gpu 2
        
        done
      done
    done
  done
done
