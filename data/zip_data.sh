current_dir=$(basename $(pwd))
if [ $current_dir = "data" ]
then
    zip spells.zip *.csv   
else
    zip data/spells.zip data/*.csv
fi
