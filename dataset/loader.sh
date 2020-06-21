# Download 300W-LP
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B7OEHD3T4eCkVGs0TkhUWFN6N1k" -O 300W-LP.zip && rm -rf /tmp/cookies.txt
# Download AFLW2000
wget -c http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/Database/AFLW2000-3D.zip -O AFLW2000.zip

# Unzip 300W-LP dataset
unzip 300W-LP.zip
# Unzip AFLW2000 dataset
unzip AFLW2000.zip

# Prepare cropped images from matrices
python3 extract_pose.py --db './300W_LP/AFW' --output './AFW.npz'
python3 extract_pose.py --db './300W_LP/AFW_Flip' --output './AFW_Flip.npz'
python3 extract_pose.py --db './300W_LP/HELEN' --output './HELEN.npz'
python3 extract_pose.py --db './300W_LP/HELEN_Flip' --output './HELEN_Flip.npz'
python3 extract_pose.py --db './300W_LP/IBUG' --output './IBUG.npz'
python3 extract_pose.py --db './300W_LP/IBUG_Flip' --output './IBUG_Flip.npz'
python3 extract_pose.py --db './300W_LP/LFPW' --output './LFPW.npz'
python3 extract_pose.py --db './300W_LP/LFPW_Flip' --output './LFPW_Flip.npz'

python3 extract_pose.py --db './AFLW2000' --output './AFLW2000.npz'

rm -r 300W_LP/
rm -r AFLW2000/

mkdir 300W_LP
mkdir AFLW2000

python3 prepare_dataset.py --path './300W_LP' --dbs 'AFW.npz' 'AFW_Flip.npz' --output 'pose_300w.csv'
python3 prepare_dataset.py --path './AFLW2000' --dbs 'AFLW2000.npz' --output 'pose_aflw.csv'
python3 split_dataset.py

# Delete remained numpy arrays
rm -r './AFW.npz'
rm -r './AFW_Flip.npz'
rm -r './HELEN.npz'
rm -r './HELEN_Flip.npz'
rm -r './IBUG.npz'
rm -r './IBUG_Flip.npz'
rm -r './LFPW.npz'
rm -r './LFPW_Flip.npz'
rm -r './AFLW2000.npz'

# Rename config file
mv ../data/rename_to_config.json ../data/config.json

