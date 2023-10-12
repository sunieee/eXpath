

# dataset=MOF-3000
# method=ConvE          TransE
# embedding_model=CompGCN       ""

explain() {
        dataset=$1
        embedding_model=""
        method=$2
        run=$3  # 111
        system=$4
        device=$5

        name="${method}${embedding_model}_${dataset}-${system}"
        dirname=stage7/${name}
        echo $dirname
        rm -rf $dirname
        if [[ $method == "ComplEx" ]]; then
                process=3
        else
                process=6
        fi
        echo "process: $process"

        for i in $(seq 1 $process); do
                output_folder=${dirname}/${i}

                echo $output_folder
                mkdir -p $output_folder
                CUDA_VISIBLE_DEVICES=$device python explain.py --dataset $dataset --method $method --system $system \
                        --run $run --relevance_method score --output_folder $output_folder --process $process --split $i \
                        2>&1 > $output_folder/output.log &
                        # --specify_relation  \
                        # --embedding_model "$embedding_model" --train_restrain
                pids[${#pids[@]}]=$! # Store PID of the last background process
                sleep 0.5
        done
}


explain_all() {
        dataset=$1
        method=$2
        echo "Explain all system for $dataset $method"

        pids=()
        
        explain $dataset $method 00011 data_poisoning 0
        explain $dataset $method 00011 k1 1
        explain $dataset $method 00011 kelpie 2
        if [[ $method != "TransE" ]]; then
                explain $dataset $method 00011 criage 3
        fi               

        # Wait for all background processes to complete
        for pid in ${pids[*]}; do
                wait $pid
        done
}


# explain_all MOF-3000 ConvE
# for method in ConvE TransE ComplEx; 


for method in ConvE ComplEx;do
        for dataset in WN18RR FB15k-237 MOF-3000;do
                explain_all $dataset $method
        done
done



# explain CompGCN ConvE 011
# explain CompGCN ConvE 011


# explain FB15k-237 ComplEx 00011 criage 0
# explain_all MOF-3000 ComplEx