#!/usr/bin/env python3
import os
import random
import torch
import numpy as np
from sklearn.svm import SVC

# ------------------------------------------------------------------
# Utility: parse subject + image from 'original_pair'
# ------------------------------------------------------------------
def parse_subject_image(pair_info):
    """
    pair_info is something like:
      (
        'subject_0_image_n02492035_gran_coarse_coeffs.txt',
        'subject_0_image_n02492035_gran_coarse_channels.txt'
      )
    We just look at the coeffs filename, remove '_coeffs.txt',
    then split on '_image_'
    """
    coeffs_filename = pair_info[0]  # e.g. "subject_0_image_n02492035_gran_coarse_coeffs.txt"
    base = coeffs_filename.replace('_coeffs.txt', '')
    parts = base.split('_image_')  # e.g. ["subject_0", "n02492035_gran_coarse"]
    if len(parts) != 2:
        # fallback
        return None, None
    subject_str = parts[0]       # "subject_0"
    image_str = parts[1]         # "n02492035_gran_coarse"
    return subject_str, image_str

# ------------------------------------------------------------------
# Utility: feature extraction (bag-of-tokens)
# ------------------------------------------------------------------
def extract_feature_vector(prompt_tokens, completion_tokens, max_vocab=5000):
    """
    Creates a simplistic bag-of-tokens frequency vector of length 2*max_vocab:
      first half for prompt, second half for completion.
    This is purely a demonstration approach.
    """
    # 1) Truncate token IDs above max_vocab
    prompt_ids = [min(int(t.item()), max_vocab - 1) for t in prompt_tokens]
    completion_ids = [min(int(t.item()), max_vocab - 1) for t in completion_tokens]

    # 2) Build frequency histograms
    hist_prompt = np.zeros(max_vocab, dtype=np.float32)
    hist_completion = np.zeros(max_vocab, dtype=np.float32)

    for pid in prompt_ids:
        hist_prompt[pid] += 1
    for cid in completion_ids:
        hist_completion[cid] += 1

    # 3) Concatenate
    feat_vec = np.concatenate([hist_prompt, hist_completion], axis=0)  # shape (2*max_vocab,)
    return feat_vec

# ------------------------------------------------------------------
# Build (X, y) for a single subject
# ------------------------------------------------------------------
def build_dataset_for_subject(
    subject_data,    # { image_id: [shard_dict, shard_dict, ...], ... }
    segment_size=128,
    max_pairs=2000,
    max_vocab=5000
):
    """
    subject_data: a dict mapping image_id -> list of shard dicts.
                  Each shard dict has: { 'tokens': Tensor, 'channels': Tensor, ... }
    We create correct/wrong pairs *within this one subject*:
      - correct: (prompt, next chunk) from the same image.
      - wrong:   (prompt) from one image, (completion) from a different image.
    We generate up to 'max_pairs' pairs in total. Half correct, half wrong, or approximate.

    Returns:
      X (numpy array), y (0/1 labels), each shaped (#pairs, feature_dim).
    """
    image_ids = list(subject_data.keys())
    if len(image_ids) < 2:
        # Not enough images to build "wrong" pairs from a different image
        return None, None

    X_list = []
    y_list = []

    num_created = 0
    # We'll just keep looping until we hit max_pairs (or run out of options)
    while num_created < max_pairs:
        # 1) Pick a random image for the correct pair
        img_id_correct = random.choice(image_ids)
        shards_correct = subject_data[img_id_correct]
        shard_c = random.choice(shards_correct)
        tokens_c = shard_c['tokens']  # shape (N,)
        N = tokens_c.size(0)
        if N < 2*segment_size:
            # not enough tokens for a correct pair
            continue

        # pick a random start
        start_idx = random.randint(0, N - 2*segment_size)
        prompt_tokens = tokens_c[start_idx : start_idx + segment_size]
        correct_tokens = tokens_c[start_idx + segment_size : start_idx + 2*segment_size]

        # 2) Build a correct example
        feat_correct = extract_feature_vector(prompt_tokens, correct_tokens, max_vocab=max_vocab)
        X_list.append(feat_correct)
        y_list.append(1)
        num_created += 1

        # 3) Build a wrong example
        # pick a different image from the same subject
        other_image_ids = [iid for iid in image_ids if iid != img_id_correct]
        if not other_image_ids:
            # can't build a wrong example if only one image for subject
            continue

        img_id_wrong = random.choice(other_image_ids)
        shards_wrong = subject_data[img_id_wrong]
        shard_w = random.choice(shards_wrong)
        tokens_w = shard_w['tokens']
        if tokens_w.size(0) < segment_size:
            continue
        max_start_w = tokens_w.size(0) - segment_size
        start_idx_w = random.randint(0, max_start_w)
        wrong_tokens = tokens_w[start_idx_w : start_idx_w + segment_size]

        feat_wrong = extract_feature_vector(prompt_tokens, wrong_tokens, max_vocab=max_vocab)
        X_list.append(feat_wrong)
        y_list.append(0)
        num_created += 1

    if not X_list:
        return None, None

    X = np.stack(X_list, axis=0)  # shape (#pairs, 2*max_vocab)
    y = np.array(y_list, dtype=np.int64)
    return X, y


# ------------------------------------------------------------------
# Evaluate forced choice *within a subject*, using the trained SVM
# ------------------------------------------------------------------
def evaluate_subject_forced_choice_svm(
    subject_data,
    svm_model,
    segment_size=128,
    max_vocab=5000,
    num_trials=50
):
    """
    For each trial:
      - pick a random (prompt) from one image (call it "correct" image),
      - pick correct completion from that same image,
      - pick several wrong completions from the *other images* of the same subject,
      - see if the correct completion yields the highest SVM decision_function.
    Returns the fraction of correct forced-choice picks.
    """
    image_ids = list(subject_data.keys())
    if len(image_ids) < 2:
        print("  [evaluate_subject_forced_choice_svm] Only 1 image for subject => skip.")
        return None

    total_count = 0
    correct_count = 0

    for _ in range(num_trials):
        # pick a "correct" image
        img_id_correct = random.choice(image_ids)
        shards_correct = subject_data[img_id_correct]
        shard_c = random.choice(shards_correct)
        tokens_c = shard_c['tokens']
        N = tokens_c.size(0)
        if N < 2*segment_size:
            # skip
            continue
        start_idx = random.randint(0, N - 2*segment_size)
        prompt_tokens = tokens_c[start_idx : start_idx + segment_size]
        correct_tokens = tokens_c[start_idx + segment_size : start_idx + 2*segment_size]

        # gather completions
        completions = []
        completions.append(('correct', correct_tokens))

        # now pick "wrong" completions from other images
        other_image_ids = [iid for iid in image_ids if iid != img_id_correct]
        # let's pick e.g. 3 wrong completions
        wrong_count = 0
        for _w in range(3):
            img_id_w = random.choice(other_image_ids)
            shards_w = subject_data[img_id_w]
            shard_w = random.choice(shards_w)
            tokens_w = shard_w['tokens']
            if tokens_w.size(0) < segment_size:
                continue
            max_start_w = tokens_w.size(0) - segment_size
            stw = random.randint(0, max_start_w)
            wcomp = tokens_w[stw : stw + segment_size]
            completions.append(('wrong', wcomp))
            wrong_count += 1

        if wrong_count == 0:
            # no valid wrong completions
            continue

        # SVM decision among these completions
        best_score = -1e9
        best_label = None

        for label_str, comp_tokens in completions:
            feat_vec = extract_feature_vector(prompt_tokens, comp_tokens, max_vocab=max_vocab)
            feat_vec = feat_vec.reshape(1, -1)

            score = svm_model.decision_function(feat_vec)[0]
            # For a binary SVC, 'decision_function' returns a single float
            # > 0 => more likely label=1, < 0 => more likely label=0

            if score > best_score:
                best_score = score
                best_label = label_str

        total_count += 1
        if best_label == 'correct':
            correct_count += 1

    if total_count == 0:
        return None
    return correct_count / total_count


# ------------------------------------------------------------------
# Main function: train & evaluate subject-aware
# ------------------------------------------------------------------
def run_svm_subject_aware(
    shards_dir="validation_datasets_imageNet/shards",
    segment_size=128,
    max_pairs=1000,
    num_trials=50,
    test_ratio=0.2,
    max_vocab=5000
):
    """
    1) Load all .pt shards & parse subject, image
    2) Build data structure: data_by_subject
    3) For each subject:
       - Build (X, y) from that subject's shards
       - Split into train/test
       - Train an SVM for that subject
       - Evaluate multi-class forced choice on the test portion
    4) Average all-subjects accuracy
    """
    # 1) Collect shards
    pt_files = [
        f for f in os.listdir(shards_dir)
        if f.endswith(".pt") and "shard_train_" in f
    ]
    if not pt_files:
        print(f"No .pt found in {shards_dir}.")
        return

    data_by_subject = {}

    for ptf in pt_files:
        full_path = os.path.join(shards_dir, ptf)
        shard_data = torch.load(full_path)
        pair_info = shard_data['original_pair']
        subj_id, img_id = parse_subject_image(pair_info)
        if (subj_id is None) or (img_id is None):
            # skip invalid
            continue
        if subj_id not in data_by_subject:
            data_by_subject[subj_id] = {}
        if img_id not in data_by_subject[subj_id]:
            data_by_subject[subj_id][img_id] = []
        data_by_subject[subj_id][img_id].append(shard_data)

    # 2) For each subject, build dataset, train SVM, evaluate
    subject_accuracies = []
    subjects = list(data_by_subject.keys())
    print(f"Found {len(subjects)} subjects: {subjects}")

    for subject in subjects:
        images_dict = data_by_subject[subject]
        print(f"\n=== Subject: {subject}, #images={len(images_dict)}")

        # a) Build dataset
        X, y = build_dataset_for_subject(
            images_dict,
            segment_size=segment_size,
            max_pairs=max_pairs,
            max_vocab=max_vocab
        )
        if (X is None) or (len(X) < 10):
            print(f"  Not enough data for subject {subject}, skipping.")
            continue

        # b) Train/Test Split
        # We'll do a simple random shuffle -> split
        idxs = np.arange(len(X))
        np.random.shuffle(idxs)
        split = int(len(X)*(1.0 - test_ratio))
        train_idxs = idxs[:split]
        test_idxs  = idxs[split:]

        X_train = X[train_idxs]
        y_train = y[train_idxs]
        X_test  = X[test_idxs]
        y_test  = y[test_idxs]

        print(f"  Dataset for subject {subject}: total={len(X)}, train={len(X_train)}, test={len(X_test)}")

        if len(X_train) < 2 or len(X_test) < 2:
            print(f"  Not enough train/test for subject {subject}, skipping.")
            continue

        # c) Train SVM
        svm_model = SVC(kernel='rbf', decision_function_shape='ovr')
        svm_model.fit(X_train, y_train)

        # d) Evaluate forced choice on test set
        #    Instead of standard classification, we'll do the multi-class forced choice approach,
        #    but we only have shards in `subject_data`. We'll do an actual forced-choice routine
        #    that picks random prompt+completion from the subject data,
        #    then uses the trained SVM to rank them.
        print(f"  Evaluating forced choice for subject={subject} ...")
        acc_subject = evaluate_subject_forced_choice_svm(
            subject_data=images_dict,
            svm_model=svm_model,
            segment_size=segment_size,
            max_vocab=max_vocab,
            num_trials=num_trials
        )
        if acc_subject is None:
            print(f"  No forced choice trials were possible for subject {subject}.")
            continue

        print(f"  -> Subject {subject} accuracy = {acc_subject:.4f}")
        subject_accuracies.append(acc_subject)

    # 3) Overall Accuracy
    if not subject_accuracies:
        print("No subject accuracies available. Done.")
        return
    overall_acc = sum(subject_accuracies) / len(subject_accuracies)
    print(f"\n=== Final Overall Accuracy (averaged over subjects) = {overall_acc:.4f}")


# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
if __name__ == "__main__":
    run_svm_subject_aware(
        shards_dir="validation_datasets_imageNet/shards",
        segment_size=512,
        max_pairs=1000,  # how many correct+wrong pairs to gather per subject
        num_trials=50,   # how many forced-choice trials per subject
        test_ratio=0.2,  # 80% train, 20% test
        max_vocab=5000
    )
