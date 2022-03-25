LR=0.1
LE=1
EP=1000

for RUN in 2; do

    # threshold 5 => 0
    for ALGO in 3; do
        for FRAC in 0.1 0.05; do
        
        
        # shard 10 fed w/ sel
        python3 main_slct_seq.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
        --num_users 100 --shard_per_user 10 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save cossim${RUN} --myalgo ${ALGO} --gamma 1.6 --wndw_size 20
        
    
        # shard 2 fed w/ sel
        python3 main_slct_seq.py --dataset cifar10 --model cnn --num_classes 10 --epochs ${EP} --lr ${LR} \
        --num_users 100 --shard_per_user 2 --frac ${FRAC} --local_ep ${LE} --local_bs 50 \
        --results_save cossim${RUN} --myalgo ${ALGO} --gamma 0.8 --wndw_size 20


        done
    done
done