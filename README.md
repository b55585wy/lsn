# BioSleepXSeq Model Submission

This package contains the necessary code to reproduce the experiments using the `BioSleepXSeq` model, as configured by `config_base.json`.

## Directory Structure

- `config_base.json`: The main configuration file for the experiment.
- `train_Kfold_CV.py`: The main script to run the k-fold cross-validation training.
- `parse_config.py`: Utility for parsing the configuration file.
- `base/`: Contains base classes for the trainer.
- `data_loader/`: Contains the data loading scripts.
- `model/`: Contains the model architecture and its components (`BioSleepXSeq`, losses, metrics, etc.).
- `model_mamba/`: Contains the Mamba implementation.
- `trainer/`: Contains the `Trainer` class.
- `utils/`: Contains various utility functions.

## How to Run

1.  **Prerequisites**:
    *   Ensure you have a Python environment with all necessary libraries installed (e.g., PyTorch, NumPy).
    *   Place your preprocessed data in a directory.

2.  **Execute the Training Script**:
    Run the `train_Kfold_CV.py` script with the appropriate arguments. The script requires the path to the configuration file, the data directory, and the fold ID to run.

    You can run a single fold using a command like this:

    ```bash
    python train_Kfold_CV.py -c config_base.json --np_data_dir /path/to/your/data --fold_id 0
    ```

    To run all folds, you can use a simple shell loop:

    ```bash
    for i in {0..19}; do
      python train_Kfold_CV.py -c config_base.json --np_data_dir /path/to/your/data --fold_id $i
    done
    ```

    Replace `/path/to/your/data` with the actual path to your dataset.