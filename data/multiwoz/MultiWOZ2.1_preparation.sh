wget https://github.com/xiami2019/MultiWOZ_Datasets/blob/main/MultiWOZ_2_1.zip
unzip MultiWOZ_2_1.zip
cd ./utils
python data_analysis.py --version 2.1
python preprocess2_1.py 
python postprocessing_dataset.py --version 2.1
cd ..
cp special_token_list.txt ./MultiWOZ_2.0/multi-woz-fine-processed/special_token_list.txt