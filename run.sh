#################################
# 7-24
# Write sequence 00 first 200 frames to disk
SAVE_DIR=results/7-24/v1/seq00/images # session-1.0
SEQ_NUM='00'
NUM_FRAMES=20
python vis_kitti_seq.py --save_dir $SAVE_DIR --sequence_number $SEQ_NUM --num_frames $NUM_FRAMES
