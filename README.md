# Required Libraries
    pip install matplotlib mne mne-bids mne-icalabel autoreject scipy numpy pandas


# What is happening

- utils.py: all functions used by the jupyter notebooks
    - preprocess_subject: take one subject id and preprocess it (filter frequencies)
        - returns a raw instance
    - run_ica: Run ICA for the given raw instance, then do the final frequency filter (0.1 - 30Hz)
    - create_evokeds: takes one raw instance
        - create epochs for feedback and cue win/loss
        - reject peak-to-peak 120µV
        - Additionally, use autoreject to remove bad epochs
        - plot topographies and frequencies
        - Write average epochs and evokeds to disk for processing in 02-analysis.ipynb
    - create_subplot_all_subjects:
        - Used by 02-analysis.ipynb to show the grand averages per single subject in a single plot

- 01-preprocessing.ipynb:
    - Run preprocessing for all subjects (uses utils.py functions)
        - So that all files get created for further processing

- 02-analysis.ipynb:
    - Read all files that were created by eeg-preprocessing
    - Then create the grand average of all 12 subjects
    - Finally, create the graphs

- 03-linear-regression.ipynb:
    - Read the epochs created in 01-preprocessing.ipynb and train multiple linear models