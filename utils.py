#
# Functions that will be used in our Jupyter notebooks
#

import os
import mne
from mne_icalabel import label_components
from mne_bids import (BIDSPath, read_raw_bids)
from autoreject import get_rejection_threshold

def preprocess_subject(bids_root, subject_id):
    """
    Preprocess a subject: ICA, lowpass
    Returns a raw instance
    """

    bids_path = BIDSPath(subject=subject_id,task="casinos",
                        datatype='eeg', suffix='eeg',
                        root=bids_root)
    
    # read the file
    raw = read_raw_bids(bids_path)

    print(raw)

    # Set the montage so we can create topo maps, as the data does not include them
    easycap_montage = mne.channels.make_standard_montage("easycap-M1")
    raw.set_montage(easycap_montage)

    raw.resample(250, npad="auto")

    raw.compute_psd().plot()

    raw.load_data()

    # Prefiltering for ICA
    raw.filter(l_freq=0.1, h_freq=100, method='fir', fir_design='firwin')
    notch_filter_freq = [50]
    raw.notch_filter(freqs=notch_filter_freq, method='fir', fir_design='firwin')

    # Plot after prefiltering
    raw.compute_psd().plot()
    raw.compute_psd().plot_topomap();

    return raw


def run_ica(raw):
    """
    Run ICA for a given raw data
    Modifies the raw instance in place with the fitted ICA
    """

    # Requirement for MNE-ICLabel
    ica_raw = raw.copy().filter(l_freq=1.0, h_freq=100.0)
    ica_raw = ica_raw.set_eeg_reference('average')
    
    # Run ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter="auto", method='infomax', fit_params=dict(extended=True))
    ica.fit(ica_raw)

    ic_labels = label_components(ica_raw, ica, method='iclabel')
    exclude_idx = [
        idx for idx, label in enumerate(ic_labels["labels"]) if label not in ["brain", "other"]
    ]
    
    for exclude in exclude_idx:
        explained_var_ratio = ica.get_explained_variance_ratio(
            ica_raw, components=exclude, ch_type="eeg"
        )

        ratio_percent = round(100 * explained_var_ratio["eeg"])
        print(
            f"Fraction of variance in EEG signal explained by component {exclude}: "
            f"{ratio_percent}%"
        )

    explained_var_ratio = ica.get_explained_variance_ratio(ica_raw, ch_type="eeg")
    for channel_type, ratio in explained_var_ratio.items():
        print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

    print(f"Excluding these ICA components: {exclude_idx}")
    print([f"{idx}: {ic_labels['labels'][idx]}" for idx in exclude_idx])

    ica.exclude = exclude_idx

    ica.plot_components()
    ica.plot_overlay(raw, exclude=exclude_idx);

    ica.apply(raw, exclude=exclude_idx)

    # Lowpass filter after ICA
    raw.filter(l_freq=0.1, h_freq=30, method='fir', fir_design='firwin')
    raw.set_eeg_reference(["TP9", "TP10"])

    # noisy_chs = mne.preprocessing.find_bad_channels_lof(raw)
    # print("Removing noisy channels:", noisy_chs)
    # raw.info["bads"].extend(noisy_chs)

    raw.compute_psd().plot()

    return raw


def create_evokeds(raw, subject_id):
    """
    First, create the epochs
    Then, drop bad epochs automatically
    Finally, write the evokeds into files
    """

    # Epoching
    events, event_id = mne.events_from_annotations(raw)

    tmin, tmax = -0.2, 0.6  # Time range in seconds
    
    reject_criteria = dict(eeg=120e-6)  # Add 120 µV threshold for EEG channels

    events_relevant = {
        # win events
        "S  6": "win (low-task, low-cue)",
        "S 16": "win (mid-task, low-cue)",
        "S 26": "win (mid-task, high-cue)",
        "S 36": "win (high-task, high-cue)",

        # loss_events
        "S  7": "loss (low-task, low-cue)",
        "S 17": "loss (mid-task, low-cue)",
        "S 27": "loss (mid-task, high-cue)",
        "S 37": "loss (high-task, high-cue)"
    }

    # All processed epochs and evokeds files are saved in the folder fif-files
    if not os.path.isdir("fif-files"):
        os.mkdir("fif-files")

    # For all win and loss events, get epochs, reject bad epochs and save the evokeds
    for key, event_type in events_relevant.items():
        epochs = mne.Epochs(
            raw, events, event_id=event_id[f"Stimulus/{key}"], tmin=tmin, tmax=tmax,
            baseline=(None, 0), reject=reject_criteria, preload=True, reject_by_annotation=True
        )

        reject = get_rejection_threshold(epochs, decim=2)
        print(f"{event_type} epochs rejection threshold for {subject_id} is {reject['eeg']}")
        epochs.drop_bad(reject=reject)
        evokeds = epochs.average()

        epochs.save(f"fif-files/subject{subject_id}_feedback_{event_type.replace(' ', '-')}-epo.fif", overwrite=True)
        evokeds.save(f"fif-files/subject{subject_id}_feedback_{event_type.replace(' ', '-')}-ave.fif", overwrite=True)


def create_subplot_all_subjects(event, subject_ids, plt):
    """
    Create a 3x4 or 2x4 ERP plot of all subjects
    """

    nrows = 3
    ncols = 4
    if len(subject_ids) == 8:
        nrows = 2

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(12, 6)
    )

    fig.subplots_adjust(hspace=0.4)

    for i, subject_id in enumerate(subject_ids):
        evoked = mne.read_evokeds(f"fif-files/subject{subject_id}_feedback_{event.replace(' ', '-')}-ave.fif")
        evoked[0].set_eeg_reference(["TP9", "TP10"])
        
        row = i // ncols
        col = i % ncols
        axes[row, col].plot(evoked[0].get_data(picks="FCz")[0])
        axes[row, col].set_title(f"Subject {subject_id}")

    plt.suptitle(event + " for all subjects (FCz)")
    plt.show()