#export NCCL_IB_DISABLE=1
python3 testing.py  --num_gpus 4                        \
 --coco_path /tmp/pycharm_project_DETR                  \
 --num_queries 1000  --two_stage                        \
 --batch_size 1 --start_epoch 35 --end_epoch 50    --enc_layers 6 --dec_layers 3     \
 --with_box_refine  --aps 2 --AMself 0 --DeA 0 --ARelation 1 --output_dir two-stage-deformable-ADis-real-3-2-ID_token
python3 demo_opt.py record-final 35
