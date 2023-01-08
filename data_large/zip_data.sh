current_dir=$(basename $(pwd))
if [ $current_dir = "data_large" ]
then
    zip spells.zip *.csv   
else
    zip data_large/spells.zip data_large/*.csv
fi
