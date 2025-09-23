export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

declare -a datasets=(
  "DSB2018 ./data/DSB2018 ./data/MonuSeg2018-test MonuSeg2018-test"
  "busi ./data/busi ./data/bus bus"
  "DRIVE ./data/DRIVE ./data/CHASEDB1 CHASEDB1"
  "isic18 ./data/isic18 ./data/PH2-test PH2-test"
  "Kvasir-SEG ./data/Kvasir-SEG ./data/Kvasir-test cvc300"
  "Glas ./data/Glas"  # no zero-shot
  "tuscui ./data/tuscui ./data/TUCC-test/malignant malignant"
  "Montgomery ./data/Montgomery ./data/NIH-test NIH-test"
  "BUSBRA ./data/BUSBRA ./data/bus bus"
)

for entry in "${datasets[@]}"; do
  set -- $entry
  dataset_name=$1
  base_dir=$2
  zero_base_dir=$3
  zero_name=$4

  cmd="python main.py --max_epochs 300 --gpu 0 --batch_size 8 --model AC_MambaSeg --base_dir $base_dir --dataset_name $dataset_name  --just_for_test True"
  if [ -n "$zero_base_dir" ]; then
    cmd+=" --zero_shot_base_dir $zero_base_dir --zero_shot_dataset_name $zero_name"
  fi

  echo "Running: $cmd"
  eval $cmd
done