export CUDA_VISIBLE_DEVICES='1'

#python generate.py --algor ifgsm --model_names vgg16 googlenet resnet20_cifar100 resnet1001_cifar100 preresnet20_cifar100 preresnet1001_cifar100 seresnet20_cifar100 seresnet272bn_cifar100 densenet40_k12_cifar100 densenet250_k24_bc_cifar100 pyramidnet110_a84_cifar100 pyramidnet272_a200_bn_cifar100 resnext29_32x4d_cifar100 wrn28_10_cifar100 nin_cifar100 ror3_164_cifar100 --max_iter 20 --lr 0.2

#python generate.py --algor ifgsm --model_names vgg16 googlenet resnet110_cifar100 preresnet110_cifar100 seresnet110_cifar100 densenet100_k12_bc_cifar100 pyramidnet110_a84_cifar100 resnext29_32x4d_cifar100 nin_cifar100 ror3_164_cifar100 --max_iter 20 --out_dir adv_image_20_01 --lr 0.1
#python generate.py --algor ifgsm --model_names vgg16 googlenet resnet110_cifar100 preresnet110_cifar100 seresnet110_cifar100 densenet100_k12_bc_cifar100 pyramidnet110_a84_cifar100 resnext29_32x4d_cifar100 nin_cifar100 ror3_164_cifar100 --max_iter 20 --out_dir adv_image_20_02 --lr 0.2
#python generate.py --algor ifgsm --model_names vgg16 googlenet resnet110_cifar100 preresnet110_cifar100 seresnet110_cifar100 densenet100_k12_bc_cifar100 pyramidnet110_a84_cifar100 resnext29_32x4d_cifar100 nin_cifar100 ror3_164_cifar100 --max_iter 20 --out_dir adv_image_20_05 --lr 0.5

#python generate.py --algor ifgsm --model_names vgg16 googlenet resnet110_cifar100 preresnet110_cifar100 seresnet110_cifar100 densenet100_k12_bc_cifar100 pyramidnet110_a84_cifar100 resnext29_32x4d_cifar100 nin_cifar100 ror3_164_cifar100 --max_iter 10 --out_dir adv_images_10
#python generate.py --algor ifgsm --model_names vgg16 googlenet resnet110_cifar100 preresnet110_cifar100 seresnet110_cifar100 densenet100_k12_bc_cifar100 pyramidnet110_a84_cifar100 resnext29_32x4d_cifar100 nin_cifar100 ror3_164_cifar100 --max_iter 20 --out_dir adv_images_20
#python generate.py --algor ifgsm --model_names vgg16 googlenet resnet110_cifar100 preresnet110_cifar100 seresnet110_cifar100 densenet100_k12_bc_cifar100 pyramidnet110_a84_cifar100 resnext29_32x4d_cifar100 nin_cifar100 ror3_164_cifar100 --max_iter 50 --out_dir adv_images_50
#python generate.py --algor ifgsm --model_names vgg16 googlenet resnet110_cifar100 preresnet110_cifar100 seresnet110_cifar100 densenet100_k12_bc_cifar100 pyramidnet110_a84_cifar100 resnext29_32x4d_cifar100 nin_cifar100 ror3_164_cifar100 --max_iter 100 --out_dir adv_images_100
