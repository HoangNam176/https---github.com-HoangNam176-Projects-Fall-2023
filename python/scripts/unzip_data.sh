dir_data=/home/hoang/Documents/work/Data/Face_Anti_Spoofing

# download file celeb spoofing
for file in "$dir_data"/*
do
    7z x $file
done