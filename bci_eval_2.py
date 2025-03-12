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
from tqdm import tqdm
from collections import OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


class EEGEvaluator:
    """
    Evaluates an EEG Transformer model using few-shot learning and classifier head approaches.
    Loads tokenized data samples into RAM for faster processing.
    """

    def __init__(self, checkpoint_dir, data_dir, device="cuda", pad_token_id=129,
                 eos_token_id=128, codebook_size=130, window_size=2304,
                 d_model=32, n_heads=4, n_layers=2, max_windows=4):
        """
        Initialize the evaluator with directories and model parameters.

        Args:
            checkpoint_dir: Directory containing model checkpoints
            data_dir: Directory containing tokenized data files
            device: Device to run evaluation on (cuda/cpu)
            pad_token_id: Padding token ID used in the model
            eos_token_id: End of sequence token ID
            codebook_size: Size of the model's codebook
            window_size: Size of each EEG window
            d_model: Hidden dimension size
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_windows: Maximum number of windows per sequence
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
            tokens = torch.load(file_path, map_location="cpu")

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
        from HTETP import HierarchicalEEGTransformer  # Import from training code

        print(f"Initializing model from {checkpoint_path}...")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

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

    def _extract_embeddings(self, tokens):
        """
        Extract embeddings from the model for the given tokens

        Args:
            tokens: Token sequence [batch_size, seq_length]

        Returns:
            Embeddings tensor [batch_size, d_model]
        """
        embeddings = []

        def hook_fn(module, input, output):
            # Capture normalized representation before final projection
            embeddings.append(output.detach())

        # Register hook on normalization layer
        hook = self.model.norm.register_forward_hook(hook_fn)

        with torch.no_grad():
            # Forward pass
            self.model(tokens)

        # Remove hook
        hook.remove()

        if not embeddings:
            raise RuntimeError("Failed to capture embeddings")

        # Average over sequence length for a global representation
        avg_embedding = embeddings[0].mean(dim=1)
        return avg_embedding

    def evaluate_few_shot(self, n_shots=5, n_queries=5, n_trials=10):
        """
        Evaluate few-shot learning performance

        Args:
            n_shots: Number of examples per class for support set
            n_queries: Number of query examples per class
            n_trials: Number of random trials to average over

        Returns:
            Average accuracy across trials
        """
        print(f"Running {n_shots}-shot evaluation ({n_trials} trials)...")

        accuracies = []
        class_names = list(self.class_data.keys())

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
                    seq = sequences[idx].unsqueeze(0).to(self.device)
                    emb = self._extract_embeddings(seq)
                    support_embeddings.append(emb)
                    support_labels.append(class_idx)

                # Get embeddings for query set
                for idx in query_indices:
                    seq = sequences[idx].unsqueeze(0).to(self.device)
                    emb = self._extract_embeddings(seq)
                    query_embeddings.append(emb)
                    query_labels.append(class_idx)

            # Stack embeddings and convert labels to tensors
            support_embeddings = torch.cat(support_embeddings, dim=0)
            support_labels = torch.tensor(support_labels, device=self.device)
            query_embeddings = torch.cat(query_embeddings, dim=0)
            query_labels = torch.tensor(query_labels, device=self.device)

            # Perform nearest neighbor classification
            correct = 0
            total = len(query_labels)

            for i, query_emb in enumerate(query_embeddings):
                # Calculate distance to all support examples
                distances = torch.norm(support_embeddings - query_emb.unsqueeze(0), dim=1)

                # Find k nearest neighbors
                _, indices = torch.topk(distances, k=n_shots, largest=False)
                nearest_labels = support_labels[indices]

                # Majority vote
                votes = torch.bincount(nearest_labels, minlength=len(class_names))
                predicted_label = torch.argmax(votes)

                if predicted_label == query_labels[i]:
                    correct += 1

            # Calculate accuracy
            accuracy = correct / total if total > 0 else 0
            accuracies.append(accuracy)

            print(f"  Trial {trial + 1}/{n_trials}: Accuracy = {accuracy:.4f}")

        avg_accuracy = np.mean(accuracies)
        print(f"Few-shot ({n_shots}-shot) average accuracy: {avg_accuracy:.4f}")
        return avg_accuracy

    def evaluate_classifier(self, train_ratio=0.7, classifier_type="logistic"):
        """
        Evaluate using embeddings as features for a classifier

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
                seq_tensor = seq.unsqueeze(0).to(self.device)
                emb = self._extract_embeddings(seq_tensor)
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
        accuracy = clf.score(X_test, y_test)

        print(f"Classifier accuracy: {accuracy:.4f}")

        # Generate detailed classification report
        y_pred = clf.predict(X_test)
        print(classification_report(
            y_test, y_pred,
            target_names=[f"{cls}" for cls in self.class_to_idx.keys()]
        ))

        return accuracy

    def run_full_evaluation(self, output_dir="evaluation_results", n_shots=5):
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

        # Evaluate each checkpoint
        for ckpt_file in self.checkpoint_files:
            # Extract epoch number
            epoch_match = re.search(r'epoch_(\d+)\.pt', ckpt_file)
            if not epoch_match:
                continue

            epoch = int(epoch_match.group(1))
            print(f"\nEvaluating checkpoint from epoch {epoch}")

            # Load checkpoint
            checkpoint = torch.load(ckpt_file, map_location=self.device)

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

            # Run evaluations
            few_shot_acc = self.evaluate_few_shot(n_shots=n_shots)
            classifier_acc = self.evaluate_classifier(classifier_type="logistic")

            # Store results
            results['epoch'].append(epoch)
            results['few_shot_accuracy'].append(few_shot_acc)
            results['classifier_accuracy'].append(classifier_acc)

            # Create intermediate plot at each checkpoint
            self._create_accuracy_plot(results, output_dir, suffix=f"_checkpoint_{epoch}")

        # Create final combined plot
        self._create_accuracy_plot(results, output_dir)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, 'evaluation_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Saved results to {csv_path}")

        return results_df

    def _create_accuracy_plot(self, results, output_dir, suffix=""):
        """Create and save accuracy plot from results"""
        plt.figure(figsize=(12, 8))

        epochs = results['epoch']

        # Plot few-shot accuracy
        plt.plot(epochs, results['few_shot_accuracy'], 'b-o',
                 label='Few-shot Learning Accuracy', linewidth=2)

        # Plot classifier accuracy
        plt.plot(epochs, results['classifier_accuracy'], 'r-o',
                 label='Classifier Head Accuracy', linewidth=2)

        # Add labels and legend
        plt.title('EEG Transformer Evaluation Across Epochs', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Set y-axis limits with some padding
        max_acc = max(
            max(results['few_shot_accuracy']) if results['few_shot_accuracy'] else 0,
            max(results['classifier_accuracy']) if results['classifier_accuracy'] else 0
        )
        plt.ylim(0, min(1.0, max_acc + 0.1))

        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'accuracy_vs_epoch{suffix}.png'), dpi=300)
        plt.close()


def main():
    """Main entry point for evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate EEG Transformer model")
    parser.add_argument("--checkpoint_dir", type=str, required=True, default='checkpoints',
                        help="Directory containing model checkpoints")
    parser.add_argument("--data_dir", type=str, required=True, default='tokenized_bci_data',
                        help="Directory containing tokenized data files")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--n_shots", type=int, default=5,
                        help="Number of shots for few-shot evaluation")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on (cuda/cpu)")
    parser.add_argument("--codebook_size", type=int, default=130,
                        help="Size of the VQAE codebook")
    parser.add_argument("--window_size", type=int, default=2304,
                        help="Size of flattened EEG window (72x32)")
    parser.add_argument("--d_model", type=int, default=32,
                        help="Hidden dimension size")
    parser.add_argument("--n_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--max_windows", type=int, default=4,
                        help="Maximum number of windows per sequence")
    parser.add_argument("--pad_token_id", type=int, default=129,
                        help="Padding token ID")
    parser.add_argument("--eos_token_id", type=int, default=128,
                        help="End of sequence token ID")

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Create evaluator
    evaluator = EEGEvaluator(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        device=args.device,
        codebook_size=args.codebook_size,
        window_size=args.window_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_windows=args.max_windows,
        pad_token_id=args.pad_token_id,
        eos_token_id=args.eos_token_id
    )

    # Run full evaluation
    results = evaluator.run_full_evaluation(
        output_dir=args.output_dir,
        n_shots=args.n_shots
    )

    print("Evaluation complete!")

    # Print best results
    best_few_shot_idx = np.argmax(results['few_shot_accuracy'])
    best_classifier_idx = np.argmax(results['classifier_accuracy'])

    print("\nBest Results:")
    print(
        f"Best Few-shot accuracy: {results['few_shot_accuracy'][best_few_shot_idx]:.4f} at epoch {results['epoch'][best_few_shot_idx]}")
    print(
        f"Best Classifier accuracy: {results['classifier_accuracy'][best_classifier_idx]:.4f} at epoch {results['epoch'][best_classifier_idx]}")


if __name__ == "__main__":
    main()