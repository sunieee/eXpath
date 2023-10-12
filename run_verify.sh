# FB15k-237
# WN18
# YAGO3-10
# MOF-3000

while pgrep -f "rsync -a" > /dev/null; do
    date=`date`
    echo "[$date] Waiting for verify to finish"
    sleep 60
done


while pgrep -f "python verify" > /dev/null; do
    # echo current time
    date=`date`
    echo "[$date] Waiting for verify to finish"
    sleep 60 # 等待10秒再次检查
done



verify() {
    dataset=$1
    method=$2
    system=$3
    device=$4

    # ${s%%+*}: This is a form of parameter expansion. 
    # The %% means "remove the longest match from the end", and +* is the pattern that matches a + followed by any sequence of characters. 
    # So, this effectively splits the string at the + and takes the first part.
    output_folder=stage7/${method}_${dataset}-${system%%+*}

    echo "verify $dataset $method $system $device"
    
    for top_n in 1 5; do
        CUDA_VISIBLE_DEVICES=$device python verify.py --output_folder $output_folder \
        --dataset $dataset --method $method --process 10 --verify \
        --system $system  --top_n_explanation $top_n \
        2>&1 > $output_folder/verify_${system}_${top_n}.log &
        pids[${#pids[@]}]=$! # Store PID of the last background process
        sleep 1
    done
}


verify_all_baseline() {
    method=$1
    dataset=$2
    echo "verify_all_baseline $dataset $method"

    # Clear previous PIDs
    pids=()

    verify $dataset $method data_poisoning 0
    verify $dataset $method k1 1
    verify $dataset $method kelpie 2
    if [[ $method != "TransE" ]]; then
        verify $dataset $method criage 3
    fi

    # Wait for all background processes to complete
    for pid in ${pids[*]}; do
        wait $pid
    done
}


verify_all_xrule() {
    method=$1
    dataset=$2
    echo "verify_all_system $dataset $method"

    # Clear previous PIDs
    pids=()

    verify $dataset $method xrule 0
    verify $dataset $method k1+xrule 1
    verify $dataset $method kelpie+xrule 2

    # Wait for all background processes to complete
    for pid in ${pids[*]}; do
        wait $pid
    done
}


for method in ComplEx ConvE; do
    for dataset in FB15k-237 WN18RR MOF-3000; do
        verify_all_baseline $method $dataset
    done
done

for method in ConvE ComplEx; do
    for dataset in FB15k-237 WN18RR MOF-3000; do
        verify_all_xrule $method $dataset
    done
done

# verify_all_baseline ConvE MOF-3000

# verify_all_xrule ComplEx FB15k-237


# verify FB15k-237 ComplEx xrule 0
# verify WN18RR ComplEx kelpie 0
# verify FB15k-237 ComplEx kelpie+xrule 0
# verify FB15k-237 ConvE k1 0
# verify MOF-3000 ConvE criage 0


# verify_all_dataset ComplEx xrule

# verify MOF-3000 ConvE kelpie 1
# verify MOF-3000 ConvE k1 0

# verify WN18RR TransE k1 1
