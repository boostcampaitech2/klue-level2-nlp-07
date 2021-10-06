python train.py \
    --bsz 32 \
    --epochs 1 \
    --save_dir my_dir \
    --dev_set False

# 실행은 터미널 창에서 sh train.sh 입력
# loop 돌려서 아래와 같이 여러 실험 원큐에도 가능
# ex)
# for BSZ in 16 32 64 ;
# do
#     for EPOCHS in 5 10 ;
#     do
#         python train.py \
#             --bsz ${BSZ} \
#             --epochs ${EPOCHS}
#     done
# done
#
