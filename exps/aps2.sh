export NCCL_IB_DISABLE=1
bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
   --num_queries 1000 --epochs 50 --enc_layers 6 --dec_layers 3 \
   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 2 --AMself 0 --two_stage  \
   --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-3-2-ID_token

bash ./tools/run_dist_launch.sh 4 ./configs/r50_deformable_detr.sh   \
   --num_queries 1000 --epochs 50 --enc_layers 6 --dec_layers 3 \
   --with_box_refine  --lr_drop 40 --batch_size 2 --aps 2 --AMself 0 --two_stage  \
   --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-3-2-ID_token --resume /root/autodl-tmp/Iter-deformable-AH/DH_head/model_dump/two-stage-deformable-ADis-real-3-2-ID_token/checkpoint.pth


bash exps/aps_test2.sh
