#python main.py --dataset_path ./data/input/ --epochs 50 --lr 0.0001 --pos_percentage 1

for pos_percentage in $(seq 0 0.1 1); do
    echo "Running with pos_percentage=${pos_percentage}"
    python main.py --dataset_path ./data/input/ --epochs 50 --lr 0.0001 --pos_percentage $(echo "${pos_percentage}" | tr ',' '.')
done