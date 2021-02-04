#!/bin/bash
pip install gdown
gdown https://drive.google.com/uc?id=1FMfpuNACCcT1CzhTixYvclLGSeGMYJyo -O icra_odom.zip
apt install unzip
unzip icra_odom.zip
mkdir -p results/icra21
mv accuracy*.csv results/icra21
rm icra_odom.zip
