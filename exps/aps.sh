export NCCL_IB_DISABLE=1
bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
   --num_queries 500 --epochs 54 --enc_layers 6 --dec_layers 2 \
   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 1 --AMself 0 --two_stage  \
   --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-2-1-ONLY-ID_token-subtract-norm-addq-Giou500
#   --output_dir AMself_2_1  --start_epoch 50

#bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
#   --num_queries 500 --epochs 54 --enc_layers 6 --dec_layers 3 \
#   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 2 --AMself 0 --two_stage  \
#   --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-3-2-ONLY-ID_token-subtract-norm-addq-Giou500 --resume /root/autodl-tmp/Iter-deformable-AHA/DH_head/model_dump/two-stage-deformable-ADis-real-3-2-ONLY-ID_token-subtract-norm-addq--Giou500/checkpoint.pth
#

bash exps/aps_test.sh
