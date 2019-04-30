# sript for training info
echo "training"

# label smoothingb regularization: choose the best eps for each dataset
# adaptive angular margin: choose the best margin scale m
# feature vector regularization: evaluate the effecs of feature regularization

# three datasets: ImageNet nus-wide coco2014

python train.py --smooth 0 --m 1 --regularization 0 --dataset 'nus' --class_num 21 --savepath './models/nus/lsr0/' --logfile 'nus+lsr0.txt'
python train.py --smooth 1 --eps 0.1 --m 1 --regularization 0 --dataset nus
