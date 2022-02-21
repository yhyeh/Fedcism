LR=0.1
LE=1
EP=1000
for RUN in 1; do

:'
# iid fed
python3 main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr ${LR} \
  --num_users 100 --shard_per_user 10 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save selection${RUN} --iid

# iid fed w/ sel
python3 main_slct.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr ${LR} \
  --num_users 100 --shard_per_user 10 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save selection${RUN} --iid
'
:'
python3 main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 \
  --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 10 --local_bs 50 --results_save selection1

# shard 2 fed w/ sel
python3 main_slct.py --dataset cifar10 --model cnn --num_classes 10 --epochs 2000 --lr 0.1 \
  --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 50 --results_save selection1
'



# shard 10 fed w/ sel
python3 main_slct.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
  --num_users 100 --shard_per_user 10 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save cossim${RUN}

:'
# shard 10 fed
python3 main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
  --num_users 100 --shard_per_user 10 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save cossim${RUN}

'
:'
# shard 6 fed w/ sel
python3 main_slct.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
  --num_users 100 --shard_per_user 6 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save cossim${RUN}

# shard 6 fed
python3 main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
  --num_users 100 --shard_per_user 6 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save cossim${RUN}
'

# shard 2 fed w/ sel
python3 main_slct.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
  --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save cossim${RUN}
:'
# shard 2 fed
python3 main_fed.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
  --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep ${LE} --local_bs 50 --results_save cossim${RUN}
'

done
