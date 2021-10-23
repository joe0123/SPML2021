export CUDA_VISIBLE_DEVICES='1'
if [[ $1 == fgsm ]] || [[ $1 == ifgsm ]]
then
	python generate.py --algor $1 --model_names vgg16 googlenet resnet20_cifar100 resnet1001_cifar100 preresnet20_cifar100 preresnet1001_cifar100 seresnet20_cifar100 seresnet272bn_cifar100 densenet40_k12_cifar100 densenet250_k24_bc_cifar100 pyramidnet110_a84_cifar100 pyramidnet272_a200_bn_cifar100 resnext29_32x4d_cifar100 wrn28_10_cifar100 nin_cifar100 ror3_164_cifar100
elif [[ $1 == pgd ]]
then
	python generate.py --algor pgd --model_names vgg16 googlenet resnet20_cifar100 resnet164bn_cifar100 preresnet20_cifar100 preresnet272bn_cifar100 seresnet20_cifar100 seresnet272bn_cifar100 densenet40_k12_cifar100 densenet100_k24_cifar100 pyramidnet110_a48_cifar100 pyramidnet110_a84_cifar100 resnext29_32x4d_cifar100 wrn28_10_cifar100 nin_cifar100 ror3_164_cifar100
elif [[ $1 == opt ]]
then
	python generate.py --algor opt --model_names vgg16 googlenet resnet110_cifar100 resnet1001_cifar100 preresnet20_cifar100 preresnet1001_cifar100 seresnet110_cifar100 seresnet272bn_cifar100 densenet100_k24_cifar100 densenet250_k24_bc_cifar100 pyramidnet110_a84_cifar100 pyramidnet272_a200_bn_cifar100 resnext29_32x4d_cifar100 wrn28_10_cifar100 nin_cifar100 ror3_164_cifar100
fi
