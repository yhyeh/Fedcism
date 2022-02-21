LR=0.1
LE=1
EP=1000
for RUN in 0.2 0.4 0.8 1.6 0.1; do

# shard 10 fed w/ sel
python3 main_slct.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
  --num_users 100 --shard_per_user 10 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save gamma/${RUN} --gamma ${RUN}

# shard 2 fed w/ sel
python3 main_slct.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
  --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save gamma/${RUN}  --gamma ${RUN}

done
