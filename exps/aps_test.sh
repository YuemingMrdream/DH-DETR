export NCCL_IB_DISABLE=1
python3 testing.py  --num_gpus 4                       \
 --coco_path /tmp/pycharm_project_DETR                  \
 --num_queries 500  --two_stage                        \
 --batch_size 1 --start_epoch 40 --end_epoch 55 --enc_layers 6 --dec_layers 2    \
 --with_box_refine  --aps 1 --AMself 0 --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-2-1-ONLY-ID_token-subtract-norm-addq-Giou500
python3 demo_opt.py record-final 55
