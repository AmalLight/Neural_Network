
list_argv="$@"

list_argv=($list_argv)

echo "list arguments: ${list_argv[@]}"

len_argv=${#list_argv[@]}

len_argv=$((len_argv))

echo "length arguments: $len_argv"

out_index=$((len_argv-1))

echo "last output index: $out_index"

out=${list_argv[$out_index]}

concat=''

for i in `seq $(( len_argv-1 ))`
do
    true_i=$(( i - 1 ))

    argv_i=${list_argv[true_i]}
    
    argv_i_out=$out
    argv_i_out+=$i.ts
    
    ffmpeg -i "$argv_i" -c copy $argv_i_out
    
    concat+="$argv_i_out"
    
    if (( i < (len_argv-1) )) ; then concat+='|' ; fi
done

echo "concat files: $concat"

out+='.mp4'
echo "output file: $out"

cmd="ffmpeg -i concat:$concat -c copy $out"

echo "command: $cmd"

$cmd
