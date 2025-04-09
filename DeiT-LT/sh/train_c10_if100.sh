# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --use_env --master_port=47773 main.py \
#     --model deit_base_distilled_patch16_224 \
#     --batch-size 128 \
#     --epochs 1200 \
#     --gpu 0 \
#     --teacher-path "work/DeiT-LT/teacher/paco_sam_ckpt_cf10_if100.pth.tar" \
#     --distillation-type hard \
#     --data-path cifar10 \
#     --data-set CIFAR10LT \
#     --imb_factor 0.01 \
#     --output_dir deit_out_c10lt \
#     --student-transform 0 \
#     --teacher-transform 0 \
#     --teacher-model resnet32 \
#     --teacher-size 32 \
#     --experiment [deitlt_paco_sam_cifar10_if100] \
#     --drw 1100 \
#     --no-mixup-drw \
#     --custom_model \
#     --accum-iter 4 \
#     --save_freq 300 \
#     --weighted-distillation \
#     --moco-t 0.05 --moco-k 1024 --moco-dim 32 --feat_dim 64 --paco \
#     # --log-results \


CUDA_VISIBLE_DEVICES=0 python work/DeiT-LT/main.py \
    --model deit_base_distilled_patch16_224 \
    --batch-size 128 \
    --epochs 50 \
    --gpu 0 \
    --teacher-path "work/DeiT-LT/teacher/paco_sam_ckpt_cf10_if100.pth.tar" \
    --distillation-type hard \
    --data-path cifar10 \
    --data-set CIFAR10LT \
    --imb_factor 0.01 \
    --output_dir deit_out_c10lt \
    --student-transform 0 \
    --teacher-transform 0 \
    --teacher-model resnet32 \
    --teacher-size 32 \
    --experiment [deitlt_paco_sam_cifar10_if100] \
    --drw 25 \
    --no-mixup-drw \
    --custom_model \
    --accum-iter 4 \
    --save_freq 300 \
    --weighted-distillation \
    --moco-t 0.05 --moco-k 1024 --moco-dim 32 --feat_dim 64 --paco
