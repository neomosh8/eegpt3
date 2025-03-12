import os
import glob
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from PIL.FontFile import WIDTH
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import traceback


class EEGSimpleEvaluator:
    """
    Evaluates an EEG Transformer model using the token embeddings directly
    instead of running the full model forward pass.
    """

    def __init__(self, checkpoint_dir, data_dir, device="cuda", pad_token_id=129,
                 eos_token_id=128, codebook_size=130, window_size=2304,
                 d_model=32, n_heads=4, n_layers=2, max_windows=4):
        """
        Initialize the evaluator with directories and model parameters.
        """
        self.checkpoint_dir = checkpoint_dir
        self.data_dir = data_dir
        self.device = device
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id

        # Model parameters
        self.codebook_size = codebook_size
        self.window_size = window_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.max_windows = max_windows

        # Will be populated during loading
        self.model = None
        self.token_embedding = None  # We'll extract this from the model
        self.class_data = {}  # Dictionary mapping class names to token sequences
        self.class_to_idx = {}  # Class name to index mapping

        # Find all checkpoint files
        self.checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt")),
            key=lambda x: int(re.search(r'epoch_(\d+)\.pt', x).group(1))
        )

        if not self.checkpoint_files:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

        print(f"Found {len(self.checkpoint_files)} checkpoint files")

        # Load data into RAM
        self._load_data()

        # Initialize model with first checkpoint to get structure
        self._initialize_model(self.checkpoint_files[0])

    def evaluate_baseline(self):
        """
        Evaluate baseline accuracies based on class distribution

        Returns:
            Dictionary with baseline few-shot and classifier accuracies
        """
        print("\nEvaluating baseline model (random chance)...")

        # Calculate class distribution
        class_counts = {cls: len(seqs) for cls, seqs in self.class_data.items()}
        total_samples = sum(class_counts.values())
        class_distribution = {cls: count / total_samples for cls, count in class_counts.items()}

        print(f"Class distribution: {class_distribution}")

        # 1. Few-shot baseline: Random guessing based on class distribution
        # For few-shot learning, we'll simulate random guessing with proper class probabilities
        n_trials = 100
        np.random.seed(42)  # For reproducibility

        few_shot_accs = []
        for _ in range(n_trials):
            # Generate random predictions based on class distribution
            true_labels = []
            for cls, count in class_counts.items():
                # Use half the samples as "queries"
                true_labels.extend([self.class_to_idx[cls]] * (count // 2))

            if not true_labels:
                continue

            # Generate random predictions based on class distribution
            pred_labels = np.random.choice(
                list(self.class_to_idx.values()),
                size=len(true_labels),
                p=[class_counts[cls] / total_samples for cls in self.class_to_idx.keys()]
            )

            # Calculate accuracy
            correct = sum(p == t for p, t in zip(pred_labels, true_labels))
            acc = correct / len(true_labels) if true_labels else 0
            few_shot_accs.append(acc)

        baseline_few_shot = np.mean(few_shot_accs) if few_shot_accs else 0

        # 2. Classifier baseline: Accuracy when always predicting the most common class
        most_common_class = max(class_counts.items(), key=lambda x: x[1])[0]
        most_common_class_idx = self.class_to_idx[most_common_class]
        most_common_class_count = class_counts[most_common_class]
        baseline_classifier = most_common_class_count / total_samples

        print(f"Baseline few-shot accuracy (random guessing): {baseline_few_shot:.4f}")
        print(f"Baseline classifier accuracy (most common class: {most_common_class}): {baseline_classifier:.4f}")

        baseline_results = {
            'few_shot_accuracy': baseline_few_shot,
            'classifier_accuracy': baseline_classifier
        }

        return baseline_results
    def _load_data(self):
        """Load all tokenized data files into RAM and organize by class"""
        print("Loading tokenized data files into RAM...")

        # Load label mapping if available
        mapping_file = os.path.join(self.data_dir, "label_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                mapping_data = json.load(f)
                self.class_to_idx = mapping_data.get("label_mapping", {})
                print(f"Loaded label mapping with {len(self.class_to_idx)} classes")

        # Find all token files
        token_files = glob.glob(os.path.join(self.data_dir, "*_tokens.pt"))
        print(f"Found {len(token_files)} token files")

        # Process each file and organize by class
        for file_path in tqdm(token_files, desc="Loading data files"):
            filename = os.path.basename(file_path)

            # Extract class name from filename (format: dataset_class_tokens.pt)
            parts = filename.split('_')
            if len(parts) < 3 or parts[-1] != "tokens.pt":
                print(f"Skipping file with unclear class: {filename}")
                continue

            class_name = parts[-2]

            # Load token tensor
            tokens = torch.load(file_path, map_location="cpu",weights_only=False)

            # Store by class
            if class_name not in self.class_data:
                self.class_data[class_name] = []

            self.class_data[class_name].append(tokens)

            # Add to class mapping if not present
            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)

        # Print dataset statistics
        print(f"Loaded data for {len(self.class_data)} classes:")
        for class_name, sequences in self.class_data.items():
            print(f"  - {class_name}: {len(sequences)} sequences")

    def _initialize_model(self, checkpoint_path):
        """Initialize the model using the specified checkpoint"""
        from HTETP import HierarchicalEEGTransformer  # Import from your file

        print(f"Initializing model from {checkpoint_path}...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device,weights_only=False)

        # Create model
        self.model = HierarchicalEEGTransformer(
            codebook_size=self.codebook_size,
            window_size=self.window_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            max_windows=self.max_windows,
            pad_token_id=self.pad_token_id
        ).to(self.device)

        # Load state dict (removing 'module.' prefix if it exists from DDP training)
        state_dict = checkpoint['model_state_dict']
        if list(state_dict.keys())[0].startswith('module.'):
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            state_dict = new_state_dict

        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Extract the token embedding layer
        self.token_embedding = self.model.token_embedding

    def _extract_embeddings_from_tokens(self, tokens):
        """
        Extract embeddings directly from token IDs using the embedding layer

        Args:
            tokens: Token sequence tensor

        Returns:
            Mean token embedding vectors
        """
        # Make sure tokens are on the right device
        tokens = tokens.to(self.device)

        # Remove padding and EOS tokens for embedding extraction
        mask = (tokens != self.pad_token_id) & (tokens != self.eos_token_id)
        filtered_tokens = tokens[mask]

        # Skip empty sequences
        if filtered_tokens.numel() == 0:
            # Return zero vector
            return torch.zeros(self.d_model, device=self.device)

        # Get embeddings using the embedding layer directly
        with torch.no_grad():
            # Clamp token IDs to valid range
            filtered_tokens = torch.clamp(filtered_tokens, max=self.codebook_size - 1)
            embeddings = self.token_embedding(filtered_tokens)

        # Average embeddings across sequence length
        mean_embedding = embeddings.mean(dim=0)
        return mean_embedding

    def extract_windows_from_sequence(self, sequence, window_size=2304):
        """
        Split a tokenized sequence into individual windows

        Args:
            sequence: The full token sequence
            window_size: Size of each window (default: 2304 tokens)

        Returns:
            List of window token sequences
        """
        # Remove EOS token if present
        if sequence[-1] == self.eos_token_id:
            sequence = sequence[:-1]

        # Find all pad tokens (which separate windows)
        pad_positions = (sequence == self.pad_token_id).nonzero(as_tuple=True)[0]

        windows = []
        start_idx = 0

        # Extract windows based on pad positions
        for pad_pos in pad_positions:
            window = sequence[start_idx:pad_pos]

            # Only add windows of reasonable size
            if len(window) > 0:
                windows.append(window)

            # Next window starts after this pad
            start_idx = pad_pos + 1

        # Add the last window if there's data after the last pad
        if start_idx < len(sequence):
            window = sequence[start_idx:]
            if len(window) > 0:
                windows.append(window)

        return windows

    def evaluate_few_shot_windows(self, n_shots=5, n_queries=5, n_trials=5):
        """
        Evaluate few-shot learning at the window level

        Args:
            n_shots: Number of windows per class for support set
            n_queries: Number of windows per class for query set
            n_trials: Number of random trials to average over

        Returns:
            Average accuracy across trials
        """
        print(f"Running window-level few-shot evaluation (n_shots={n_shots}, n_queries={n_queries})...")

        # Extract windows from all sequences
        class_windows = {}
        for class_name, sequences in self.class_data.items():
            all_windows = []
            for seq in sequences:
                windows = self.extract_windows_from_sequence(seq)
                all_windows.extend(windows)

            class_windows[class_name] = all_windows
            print(f"  - {class_name}: {len(all_windows)} windows extracted")

        # Check if we have enough data and adjust parameters if needed
        min_windows = min([len(windows) for windows in class_windows.values()])

        if min_windows < n_shots + n_queries:
            adjusted_n_shots = min(n_shots, min_windows // 2)
            adjusted_n_queries = min(n_queries, min_windows - adjusted_n_shots)

            print(f"Warning: Adjusting few-shot parameters to fit dataset size.")
            print(f"Using {adjusted_n_shots}-shot with {adjusted_n_queries} queries")

            n_shots = adjusted_n_shots
            n_queries = adjusted_n_queries

        # Ensure we're using at least 1 query window
        if n_queries < 1:
            n_queries = 1
            n_shots = min(n_shots, min_windows - 1)

        accuracies = []

        # Run multiple trials
        for trial in range(n_trials):
            # Prepare support and query sets
            support_embeddings = []
            support_labels = []
            query_embeddings = []
            query_labels = []

            for class_idx, class_name in enumerate(class_windows.keys()):
                windows = class_windows[class_name]

                # Skip classes with insufficient windows
                if len(windows) < n_shots + n_queries:
                    continue

                # Randomly select support and query windows
                indices = np.random.permutation(len(windows))
                support_indices = indices[:n_shots]
                query_indices = indices[n_shots:n_shots + n_queries]

                # Get embeddings for support set
                for idx in support_indices:
                    window = windows[idx].to(self.device)
                    emb = self._extract_embeddings_from_tokens(window)
                    support_embeddings.append(emb)
                    support_labels.append(class_idx)

                # Get embeddings for query set
                for idx in query_indices:
                    window = windows[idx].to(self.device)
                    emb = self._extract_embeddings_from_tokens(window)
                    query_embeddings.append(emb)
                    query_labels.append(class_idx)

            # Skip this trial if we don't have enough data
            if not support_embeddings or not query_embeddings:
                print(f"  Trial {trial + 1}/{n_trials}: Skipped (insufficient data)")
                continue

            # Stack embeddings and convert labels to tensors
            support_embeddings = torch.stack(support_embeddings)
            support_labels = torch.tensor(support_labels, device=self.device)
            query_embeddings = torch.stack(query_embeddings)
            query_labels = torch.tensor(query_labels, device=self.device)

            # Perform nearest neighbor classification
            correct = 0
            total = len(query_labels)

            for i, query_emb in enumerate(query_embeddings):
                # Calculate distance to all support examples
                distances = torch.norm(support_embeddings - query_emb.unsqueeze(0), dim=1)

                # Find k nearest neighbors (up to n_shots)
                k = min(n_shots, len(support_labels))
                _, indices = torch.topk(distances, k=k, largest=False)
                nearest_labels = support_labels[indices]

                # Majority vote
                votes = torch.bincount(nearest_labels, minlength=len(class_windows))
                predicted_label = torch.argmax(votes)

                if predicted_label == query_labels[i]:
                    correct += 1

            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0
            accuracies.append(accuracy)

            print(f"  Trial {trial + 1}/{n_trials}: Accuracy = {accuracy:.4f} ({correct}/{total})")

        # Handle case where all trials were skipped
        if not accuracies:
            print("No valid trials completed. Please check your dataset.")
            return 0.0

        avg_accuracy = np.mean(accuracies)
        print(f"Window-level few-shot ({n_shots}-shot) average accuracy: {avg_accuracy:.4f}")
        return avg_accuracy

    def evaluate_few_shot(self, n_shots=1, n_queries=1, n_trials=5):
        """
        Evaluate few-shot learning performance using token embeddings directly

        Args:
            n_shots: Number of examples per class for support set
            n_queries: Number of examples per class for query set
            n_trials: Number of random trials to average over

        Returns:
            Average accuracy across trials
        """
        print(f"Running few-shot evaluation (n_shots={n_shots}, n_queries={n_queries})...")

        # Check if we have enough data and adjust parameters if needed
        class_names = list(self.class_data.keys())
        min_examples = min([len(self.class_data[cls]) for cls in class_names])

        # We need at least 2 examples per class (1 for support, 1 for query)
        if min_examples < 2:
            print(f"Warning: Not enough examples for few-shot (min={min_examples}). Skipping evaluation.")
            return 0.0

        # Adjust n_shots and n_queries to fit available data
        max_per_class = min_examples // 2
        adjusted_n_shots = min(n_shots, max_per_class)
        adjusted_n_queries = min(n_queries, min_examples - adjusted_n_shots)

        if adjusted_n_shots != n_shots or adjusted_n_queries != n_queries:
            print(f"Warning: Adjusting few-shot parameters to fit dataset size.")
            print(
                f"Using {adjusted_n_shots}-shot with {adjusted_n_queries} queries instead of requested {n_shots}-shot with {n_queries} queries")
            n_shots = adjusted_n_shots
            n_queries = adjusted_n_queries

        # Ensure we're using at least 1 query example
        if n_queries < 1:
            n_queries = 1
            n_shots = min_examples - 1

        accuracies = []

        # Run multiple trials
        for trial in range(n_trials):
            # Prepare support and query sets
            support_embeddings = []
            support_labels = []
            query_embeddings = []
            query_labels = []

            for class_idx, class_name in enumerate(class_names):
                sequences = self.class_data[class_name]

                # Skip classes with insufficient examples
                if len(sequences) < n_shots + n_queries:
                    continue

                # Randomly select support and query examples
                indices = np.random.permutation(len(sequences))
                support_indices = indices[:n_shots]
                query_indices = indices[n_shots:n_shots + n_queries]

                # Get embeddings for support set
                for idx in support_indices:
                    seq = sequences[idx]
                    emb = self._extract_embeddings_from_tokens(seq)
                    support_embeddings.append(emb)
                    support_labels.append(class_idx)

                # Get embeddings for query set
                for idx in query_indices:
                    seq = sequences[idx]
                    emb = self._extract_embeddings_from_tokens(seq)
                    query_embeddings.append(emb)
                    query_labels.append(class_idx)

            # Skip this trial if we don't have enough data
            if not support_embeddings or not query_embeddings:
                print(f"  Trial {trial + 1}/{n_trials}: Skipped (insufficient data)")
                continue

            # Stack embeddings and convert labels to tensors
            support_embeddings = torch.stack(support_embeddings)
            support_labels = torch.tensor(support_labels, device=self.device)
            query_embeddings = torch.stack(query_embeddings)
            query_labels = torch.tensor(query_labels, device=self.device)

            # Perform nearest neighbor classification
            correct = 0
            total = len(query_labels)

            for i, query_emb in enumerate(query_embeddings):
                # Calculate distance to all support examples
                distances = torch.norm(support_embeddings - query_emb.unsqueeze(0), dim=1)

                # Find k nearest neighbors (up to n_shots)
                k = min(n_shots, len(support_labels))
                _, indices = torch.topk(distances, k=k, largest=False)
                nearest_labels = support_labels[indices]

                # Majority vote
                votes = torch.bincount(nearest_labels, minlength=len(class_names))
                predicted_label = torch.argmax(votes)

                if predicted_label == query_labels[i]:
                    correct += 1

            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0
            accuracies.append(accuracy)

            print(f"  Trial {trial + 1}/{n_trials}: Accuracy = {accuracy:.4f} ({correct}/{total})")

        # Handle case where all trials were skipped
        if not accuracies:
            print("No valid trials completed. Please check your dataset.")
            return 0.0

        avg_accuracy = np.mean(accuracies)
        print(f"Few-shot ({n_shots}-shot) average accuracy: {avg_accuracy:.4f}")
        return avg_accuracy

    def evaluate_classifier(self, train_ratio=0.7, classifier_type="logistic"):
        """
        Evaluate using token embeddings as features for a classifier

        Args:
            train_ratio: Ratio of data to use for training
            classifier_type: Type of classifier ('logistic' or 'svm')

        Returns:
            Test accuracy
        """
        print(f"Evaluating classifier head ({classifier_type})...")

        # Prepare data
        X = []  # Embeddings
        y = []  # Labels

        for class_name, sequences in self.class_data.items():
            class_idx = self.class_to_idx[class_name]

            for seq in tqdm(sequences, desc=f"Processing {class_name}", leave=False):
                # Get embedding
                emb = self._extract_embeddings_from_tokens(seq)
                X.append(emb.cpu().numpy())
                y.append(class_idx)

        # Convert to numpy arrays
        X = np.vstack(X)
        y = np.array(y)

        # Normalize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split into train/test
        indices = np.random.permutation(len(X))
        train_size = int(len(indices) * train_ratio)
        train_idx, test_idx = indices[:train_size], indices[train_size:]

        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train classifier
        if classifier_type == "logistic":
            clf = LogisticRegression(max_iter=1000, C=1.0, solver='liblinear')
        elif classifier_type == "svm":
            clf = SVC(kernel='rbf', C=1.0)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")

        # Train and evaluate
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Classifier accuracy: {accuracy:.4f}")

        # Generate detailed classification report
        print(classification_report(
            y_test, y_pred,
            target_names=[f"{cls}" for cls in self.class_to_idx.keys()]
        ))

        return accuracy

    def run_full_evaluation(self, output_dir="evaluation_results", n_shots=1):
        """
        Run evaluation on all checkpoints and save results

        Args:
            output_dir: Directory to save evaluation results
            n_shots: Number of shots for few-shot evaluation

        Returns:
            DataFrame with evaluation results
        """
        os.makedirs(output_dir, exist_ok=True)

        # Prepare results storage
        results = {
            'epoch': [],
            'few_shot_accuracy': [],
            'classifier_accuracy': []
        }
        results['few_shot_window_accuracy'] = []

        # Check data size and print warning if needed
        class_names = list(self.class_data.keys())
        class_sizes = {cls: len(self.class_data[cls]) for cls in class_names}
        print(f"Class sizes: {class_sizes}")

        # Evaluate baseline model with random weights
        baseline_results = self.evaluate_baseline()

        # Use epoch 0 for baseline in the plot
        results['epoch'].append(0)
        results['few_shot_accuracy'].append(baseline_results['few_shot_accuracy'])
        results['classifier_accuracy'].append(baseline_results['classifier_accuracy'])

        # Evaluate each checkpoint
        for ckpt_file in self.checkpoint_files:
            # Extract epoch number
            epoch_match = re.search(r'epoch_(\d+)\.pt', ckpt_file)
            if not epoch_match:
                continue

            epoch = int(epoch_match.group(1))
            print(f"\nEvaluating checkpoint from epoch {epoch}")

            try:
                # Load checkpoint
                checkpoint = torch.load(ckpt_file, map_location=self.device,weights_only=False)

                # Update model weights
                state_dict = checkpoint['model_state_dict']
                if list(state_dict.keys())[0].startswith('module.'):
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:] if k.startswith('module.') else k
                        new_state_dict[name] = v
                    state_dict = new_state_dict

                self.model.load_state_dict(state_dict)
                self.model.eval()

                # Update token embedding reference
                self.token_embedding = self.model.token_embedding

                # Run evaluations
                few_shot_acc = self.evaluate_few_shot(n_shots=n_shots)
                classifier_acc = self.evaluate_classifier(classifier_type="logistic")
                few_shot_acc_windows = self.evaluate_few_shot_windows(n_shots=n_shots)

                # Add to your results dictionary:
                results['few_shot_window_accuracy'].append(few_shot_acc_windows)
                # Store results
                results['epoch'].append(epoch)
                results['few_shot_accuracy'].append(few_shot_acc)
                results['classifier_accuracy'].append(classifier_acc)

                # Create intermediate plot at each checkpoint
                if len(results['epoch']) > 0:
                    self._create_accuracy_plot(results, output_dir, suffix=f"_checkpoint_{epoch}")

            except Exception as e:
                print(f"Error evaluating checkpoint {ckpt_file}: {str(e)}")
                traceback.print_exc()

        # Create final combined plot if we have any results
        if len(results['epoch']) > 0:
            self._create_accuracy_plot(results, output_dir)

            # Save results to CSV
            results_df = pd.DataFrame(results)
            csv_path = os.path.join(output_dir, 'evaluation_results.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"Saved results to {csv_path}")
        else:
            print("No successful evaluations completed.")

        return pd.DataFrame(results)

    def _create_accuracy_plot(self, results, output_dir, suffix=""):
        """Create and save accuracy plot from results"""
        plt.figure(figsize=(12, 8))

        # Convert to DataFrame for easier filtering
        df = pd.DataFrame(results)

        # Separate baseline from trained models
        baseline = df[df['epoch'] == 0]
        trained = df[df['epoch'] > 0]

        # Plot trained model results
        if not trained.empty:
            plt.plot(trained['epoch'], trained['few_shot_accuracy'], 'b-o',
                     label='Few-shot Learning Accuracy', linewidth=2)
            plt.plot(trained['epoch'], trained['classifier_accuracy'], 'r-o',
                     label='Classifier Head Accuracy', linewidth=2)

        # Add baseline lines if available
        if not baseline.empty:
            baseline_few_shot = baseline['few_shot_accuracy'].iloc[0]
            baseline_clf = baseline['classifier_accuracy'].iloc[0]

            plt.axhline(y=baseline_few_shot, color='b', linestyle='--',
                        label=f'Baseline Few-shot ({baseline_few_shot:.4f})')
            plt.axhline(y=baseline_clf, color='r', linestyle='--',
                        label=f'Baseline Classifier ({baseline_clf:.4f})')

        # Add labels and legend
        plt.title('EEG Transformer Evaluation Across Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Set y-axis limits with some padding
        all_accs = list(df['few_shot_accuracy']) + list(df['classifier_accuracy'])
        if all_accs:
            max_acc = max(all_accs)
            plt.ylim(0, min(1.0, max_acc + 0.1))

        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'accuracy_vs_epoch{suffix}.png'), dpi=300)
        plt.close()


def main():
    """Main entry point for evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate EEG Transformer model")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory containing model checkpoints")
    parser.add_argument("--data_dir", type=str, default="tokenized_bci_data",
                        help="Directory containing tokenized data files")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--n_shots", type=int, default=1,
                        help="Number of shots for few-shot evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on (cuda/cpu)")
    parser.add_argument("--codebook_size", type=int, default=130,
                        help="Size of the VQAE codebook")
    parser.add_argument("--window_size", type=int, default=2304,
                        help="Size of flattened EEG window (72x32)")
    parser.add_argument("--d_model", type=int, default=32,
                        help="Hidden dimension size")

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create evaluator
    evaluator = EEGSimpleEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        device=args.device,
        codebook_size=args.codebook_size,
        window_size=args.window_size,
        d_model=args.d_model
    )

    # Run full evaluation
    results = evaluator.run_full_evaluation(
        output_dir=args.output_dir,
        n_shots=args.n_shots
    )

    print("Evaluation complete!")

    if len(results) > 0:
        # Print best results
        best_few_shot_idx = np.argmax(results['few_shot_accuracy']) if len(results['few_shot_accuracy']) > 0 else -1
        best_classifier_idx = np.argmax(results['classifier_accuracy']) if len(
            results['classifier_accuracy']) > 0 else -1

        print("\nBest Results:")
        if best_few_shot_idx >= 0:
            print(
                f"Best Few-shot accuracy: {results['few_shot_accuracy'][best_few_shot_idx]:.4f} at epoch {results['epoch'][best_few_shot_idx]}")
        if best_classifier_idx >= 0:
            print(
                f"Best Classifier accuracy: {results['classifier_accuracy'][best_classifier_idx]:.4f} at epoch {results['epoch'][best_classifier_idx]}")
    else:
        print("No successful evaluations completed.")


if __name__ == "__main__":
    main()