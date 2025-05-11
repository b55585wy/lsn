#!/bin/bash

# Define the number of folds
NUM_FOLDS=20 # Or any other number you need, e.g., 10 for 0..9

# Loop from fold 0 to NUM_FOLDS-1
for i in $(seq 0 $(($NUM_FOLDS - 1)))
do
  echo "-----------------------------------------------------"
  echo "Starting training for fold ${i}..."
  echo "-----------------------------------------------------"
  
  # Run the python script directly. The script will wait for this command to complete.
  python train_Kfold_CV.py --fold_id ${i} --device 0 --np_data_dir data20/data20npy --config config_focal_loss.json
  
  # Optional: Add a check for the exit status of the python script
  if [ $? -eq 0 ]; then
    echo "-----------------------------------------------------"
    echo "Fold ${i} completed successfully."
    echo "-----------------------------------------------------"
  else
    echo "-----------------------------------------------------"
    echo "ERROR: Fold ${i} failed. Exiting script."
    echo "-----------------------------------------------------"
    exit 1 # Exit the script if a fold fails
  fi
done

echo "All folds processed."  