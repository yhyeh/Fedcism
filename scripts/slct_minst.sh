for RUN in 1; do

:'
# iid fed
python3 main_fed.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 \
  --num_users 100 --shard_per_user 10 --frac 0.1 --local_ep 1 --local_bs 10 --results_save selection${RUN} --iid

# iid fed w/ sel
python3 main_slct.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 \
  --num_users 100 --shard_per_user 10 --frac 0.1 --local_ep 1 --local_bs 10 --results_save selection${RUN} --iid
'
# shard 10 fed
python3 main_fed.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 \
  --num_users 100 --shard_per_user 10 --frac 0.1 --local_ep 1 --local_bs 10 --results_save unbalanced${RUN}

# shard 10 fed w/ sel
python3 main_slct.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 \
  --num_users 100 --shard_per_user 10 --frac 0.1 --local_ep 1 --local_bs 10 --results_save unbalanced${RUN}
:'
# shard 6 fed
python3 main_fed.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 \
  --num_users 100 --shard_per_user 6 --frac 0.1 --local_ep 1 --local_bs 10 --results_save unbalanced${RUN}

# shard 6 fed w/ sel
python3 main_slct.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 \
  --num_users 100 --shard_per_user 6 --frac 0.1 --local_ep 1 --local_bs 10 --results_save unbalanced${RUN}

# shard 2 fed
python3 main_fed.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 \
  --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save unbalanced${RUN}

# shard 2 fed w/ sel
python3 main_slct.py --dataset mnist --model mlp --num_classes 10 --epochs 1000 --lr 0.05 \
  --num_users 100 --shard_per_user 2 --frac 0.1 --local_ep 1 --local_bs 10 --results_save unbalanced${RUN}
'
done