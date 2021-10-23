export CUDA_VISIBLE_DEVICES='0'
for model_name in resnet20_cifar100 resnet56_cifar100 \
					resnet110_cifar100 resnet164bn_cifar100 \
					resnet272bn_cifar100 resnet1001_cifar100 \
					preresnet20_cifar100 preresnet56_cifar100 \
					preresnet110_cifar100 preresnet164bn_cifar100 \
					preresnet272bn_cifar100 preresnet1001_cifar100 \
					seresnet20_cifar100 seresnet56_cifar100 \
					seresnet110_cifar100 seresnet164bn_cifar100 \
					seresnet272bn_cifar100 \
					densenet40_k12_cifar100 densenet40_k12_bc_cifar100 \
					densenet100_k12_cifar100 densenet100_k24_cifar100 \
					densenet250_k24_bc_cifar100 \
					pyramidnet110_a48_cifar100 pyramidnet110_a84_cifar100 \
					pyramidnet236_a220_bn_cifar100 pyramidnet272_a200_bn_cifar100 \
					wrn28_10_cifar100 wrn40_8_cifar100 \
					nin_cifar100
do
	echo -e "\nMODEL: "$model_name
	python generate.py --algor pgd --model_names $model_name --out_dir tmp_
	python evaluate.py --adv_dir tmp_ 
done
