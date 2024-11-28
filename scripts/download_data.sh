#!/bin/bash
echo "0 - ShapeNet NMR"
echo "1 - CUB-200"
echo "2 - Pascal3D+ Cars"
echo "3 - CompCars"
echo "4 - LSUN Horse"
echo "5 - LSUN Motorbike"
read -p "Enter the dataset ID you want to download: " ds_id

mkdir -p datasets
cd datasets


echo "start downloading CUB-200..."
wget https://data.caltech.edu/tindfiles/serve/1239ea37-e132-42ee-8c09-c383bb54e7ff/ -O CUB_200_2011.tgz
echo "done, start unzipping..."
tar -xf CUB_200_2011.tgz
mv CUB_200_2011 cub_200
echo "done"
