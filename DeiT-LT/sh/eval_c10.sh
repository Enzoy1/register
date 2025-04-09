# CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 --use_env --master_port=47773 main.py \
#     --model deit_base_distilled_patch16_224 \
#     --batch-size 256 \
#     --gpu 0 \
#     --data-path cifar10 \
#     --data-set CIFAR10LT \
#     --student-transform 0 \
#     --teacher-transform 0 \
#     --custom_model \
#     --resume "Enter checkpoint path for evaluation" \
#     --eval \

CUDA_VISIBLE_DEVICES=0 python work/DeiT-LT/main.py \
    --model deit_base_distilled_patch16_224 \
    --batch-size 256 \
    --gpu 0 \
    --data-path cifar10 \
    --data-set CIFAR10LT \
    --student-transform 0 \
    --teacher-transform 0 \
    --custom_model \
    --resume "work/deit_out_c10lt/deit_base_distilled_patch16_224_resnet32_1200_CIFAR10LT_imb50_128_[deitlt_paco_sam_cifar10_if100]/deit_base_distilled_patch16_224_resnet32_1200_CIFAR10LT_imb50_128_[deitlt_paco_sam_cifar10_if100]_best_checkpoint.pth" \
    --eval \