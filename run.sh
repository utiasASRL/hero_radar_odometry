#################################
### 7-24
## Write sequence 00 first 200 frames of images to disk
#SAVE_DIR=results/7-24/v1/seq00/images # session-1.0
#SEQ_NUM='00'
#NUM_FRAMES=20
#python save_kitti_images.py --save_dir $SAVE_DIR --sequence_number $SEQ_NUM --num_frames $NUM_FRAMES

## Write sequence 00 first 200 frames of point cloud to disk
#SAVE_DIR=results/7-24/v1/seq00/pointcloud # session-1.0
#SEQ_NUM='00'
#NUM_FRAMES=20
#python save_kitti_pointcloud.py --save_dir $SAVE_DIR --sequence_number $SEQ_NUM --num_frames $NUM_FRAMES

#################################
### 7-30
#CONFIG_PATH=config/config.json
#SAVE_DIR=results/7-30/v0/seq00/images # session-1.0
#SEQ_NUM='00'
#NUM_FRAMES=200
#python save_kitti_images.py --config $CONFIG_PATH --save_dir $SAVE_DIR --sequence_number $SEQ_NUM --num_frames $NUM_FRAMES

#CONFIG_PATH=config/config.json
#python train/dummy_trainer.py --config $CONFIG_PATH

################################
## 8-2
#CONFIG_PATH=config/config.json
#python train.py --config $CONFIG_PATH

###############################
## 8-4
#CONFIG_PATH=config/config.json
#SESSION_NAME=8-4/v0
#python train.py --config $CONFIG_PATH --session_name $SESSION_NAME

## on local PC to debug
#CONFIG_PATH=config/local_config.json
#SESSION_NAME=8-4/v1
#python train.py --config $CONFIG_PATH --session_name $SESSION_NAME

###############################
## 8-5
#CONFIG_PATH=config/config.json
#SESSION_NAME=8-5/v0
#python train.py --config $CONFIG_PATH --session_name $SESSION_NAME

##############################
## 8-6
#CONFIG_PATH=config/config.json
#SESSION_NAME=8-6/v0
#python train.py --config $CONFIG_PATH --session_name $SESSION_NAME

## on local PC to debug
#CONFIG_PATH=config/local_config.json
#SESSION_NAME=8-6/v1
#python train.py --config $CONFIG_PATH --session_name $SESSION_NAME

##############################
# 8-7
CONFIG_PATH=config/config.json
SESSION_NAME=8-7/v0
python train.py --config $CONFIG_PATH --session_name $SESSION_NAME