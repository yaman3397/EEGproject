# What is happening

- utils.py: all functions
    - preprocess_subject: take one subject id and preprocess it (filter frequencies and ICA, so blinks get removed)
        - returns a raw instance
    - create_epochs: takes one raw instance.
        - create epochs for feedback and cue win/loss
        - reject peak-to-peak 120µV (doesn't do anything?)
        - plot topographies and frequencies
        - Write average epochs to files

- 01-preprocessing:
    - Run preprocessing for all subjects (uses utils.py functions)
        - So that all files get created

- 02-analysis.ipynb:
    - Read all files that were created by eeg-preprocessing
    - Then create the grand average of all 12 subjects
    - Finally, create the graphs