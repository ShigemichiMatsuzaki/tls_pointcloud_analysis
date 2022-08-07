#DIR=/mnt/c/Users/aisl/Documents/dataset/Evo_HeliALS-TW_2021_euroSDR/
DIR=/mnt/c/Users/aisl/Documents/dataset/Evo_TLS_2021_thinned/
list=`ls $DIR`
for i in `ls $DIR`; do
    echo $i
    python3 generate_chm_from_laz.py Evo_TLS_2021_thinned\\$i
done