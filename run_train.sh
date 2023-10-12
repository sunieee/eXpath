

# dataset=MOF-3000
# method=ConvE          TransE
# embedding_model=CompGCN       ""

export CUDA_LAUNCH_BLOCKING=1

train() {
        dataset=$1
        embedding_model=""
        method=$2
        run=$3  # 111
        device=$4

        name="${method}${embedding_model}_${dataset}"
        output_folder=stage7/${name}-train
        echo $output_folder
        mkdir -p $output_folder
        CUDA_VISIBLE_DEVICES=$device python explain.py --dataset $dataset --method $method --baseline xrule \
                --run $run --relevance_method score --output_folder $output_folder  \
                2>&1 > $output_folder/evaluate.log &
                        # > $output_folder/output.log
                        # --specify_relation  \
                        # --embedding_model "$embedding_model" --train_restrains
        # CUDA_VISIBLE_DEVICES=$device python explain.py --dataset $dataset --method $method --baseline xrule \
        #         --run $run --relevance_method score --output_folder $output_folder # > $output_folder/output.log
	sleep 1
}

train_all_method() {
        dataset=$1
        device=$2

        echo "train_all_method $dataset on device $device"

        train $dataset ConvE 0110 0
        train $dataset TransE 0110 0
        train $dataset ComplEx 0110 2

        # train $dataset ConvE 1100 0
        # train $dataset TransE 1100 0
        # train $dataset ComplEx 1100 2

        # train $dataset ConvE 1100 $device
        # train $dataset TransE 1100 $device
        # train $dataset ComplEx 1100 $device

}

# train_all_method WN18RR 0
# train_all_method FB15k-237 1
# train_all_method YAGO3-10 0
# train_all_method MOF-3000 2


# train YAGO3-10 ConvE 1100 0
# train YAGO3-10 TransE 1100 1
# train YAGO3-10 ComplEx 1100 2

# train_all_method MOF-3000
# train_all_method WN18RR
# train_all_method FB15k-237

# python explain.py --dataset MOF-3000 --method ConvE --baseline xrule \
#                 --run 1100 --relevance_method score --output_folder ConvE_MOF-3000-train 

train MOF-3000 ComplEx 1100 0