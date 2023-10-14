

# dataset=MOF-3000
# method=ConvE          TransE
# embedding_model=CompGCN       ""

export CUDA_LAUNCH_BLOCKING=1

explain_file=explain.py

explain() {
        dataset=$1
        embedding_model=""
        method=$2
        # device=0
        device=`nvidia-smi --query-gpu=index --format=csv,noheader,nounits | awk '{printf "%s,", $1}' | sed 's/,$//'`

        name="${method}${embedding_model}_${dataset}-xrule"
        dirname=stage7/${name}
        echo $dirname
        rm -rf $dirname
        
        process=3
        # 将设备字符串按逗号分割为数组
        IFS=',' read -ra devices <<< "$device"
        num_devices=${#devices[@]}
        pids=()

        # 计算每个设备分配到的进程数
        process=$((process * num_devices))
        echo "all process: $process"
        
        for ((i = 0; i < process; i++)); do
                output_folder=${dirname}/$((i + 1))

                echo $output_folder
                mkdir -p $output_folder
                current_device=${devices[$((i % num_devices))]}
                CUDA_VISIBLE_DEVICES=$current_device python $explain_file --dataset $dataset --method $method --system xrule \
                        --run 00011 --relevance_method score --output_folder $output_folder --process $process --split $((i + 1))  \
                        2>&1 > $output_folder/output.log &
                pids[${#pids[@]}]=$!
                sleep 0.5
        done

        # Wait for all background processes to complete
        for pid in ${pids[*]}; do
                wait $pid
        done
        # CUDA_VISIBLE_DEVICES=$device python explain.py --dataset $dataset --method $method --system xrule \
        #         --run $run --relevance_method score --output_folder $output_folder # > $output_folder/output.log
}



# explain WN18 ConvE 0001 0
# explain FB15k-237 ConvE 0001 1
# explain MOF-3000 ConvE 0001 2

# for method in ComplEx ConvE; do
#         for dataset in FB15k-237 WN18RR MOF-3000; do
#                 explain $dataset $method
#         done
# done

explain FB15k-237 ComplEx

# explain FB15k-237 ComplEx 0,1,2

# explain MOF-3000 ConvE
# explain FB15k-237 ConvE 1