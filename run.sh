for num_pos_original in $(seq 0 50 246); do
    echo "Running with num_pos_original=${num_pos_original}"
    python main.py --evaluation_model ResNet18 --dataset_path ./data/input/ --num_pos_original $num_pos_original
done

#--num_pos_generated 10
#--root_pos_generated sdxl-1