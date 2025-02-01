import mne

# Replace with the full path to your .set file.
eeg_set_file = 'dataset/Subject2_a1.set'

# Load the EEG data.
# The function automatically finds and loads the associated .fdt file if it exists.
raw = mne.io.read_raw_eeglab(eeg_set_file, preload=True)

# Retrieve and print the channel names.
print("Channel Names:")
for ch in raw.info['ch_names']:
    print(ch)
