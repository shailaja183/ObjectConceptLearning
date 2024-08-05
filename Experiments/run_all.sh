CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/cifar10.yaml --nq 1
CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/cifar10.yaml --nq 3
CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/cifar10.yaml --nq 5

CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/cifar100.yaml --nq 1
CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/cifar100.yaml --nq 3
CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/cifar100.yaml --nq 5

CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/food101.yaml --nq 1
CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/food101.yaml --nq 3
CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/food101.yaml --nq 5

CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/dtd.yaml --nq 1
CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/dtd.yaml --nq 3
CUDA_VISIBLE_DEVICES=1 python ./main.py --config ./configs/dtd.yaml --nq 5

