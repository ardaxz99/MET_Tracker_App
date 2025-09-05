import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from datasets.dataset import WISDMDataset
from models.model import HandCraftedFeaturesExtractor, SimpleCNN1D
from models.trainer import Trainer
import numpy as np

# sklearn classifiers
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.base import clone

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def parse_args():
    parser = argparse.ArgumentParser(description="Human Activity Recognition Tracker")

    parser.add_argument("--dataset", type=str, default="WISDM",
                        help="Dataset name (currently supports: WISDM)")
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset folder")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of folds for cross-validation")
    parser.add_argument("--k_fold", type=int, default=0,
                        help="[Deprecated] Ignored when running repeated CV; test fold index (0-based)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use: cuda or cpu")
    parser.add_argument("--feature_extraction", type=str, default="raw",
                        choices=["raw", "handcrafted"],
                        help="Feature extraction method: raw (200x3 windows) or handcrafted (43 features)")
    parser.add_argument(
        "--model", type=str, default="ml",
        choices=["ml", "cnn"],
        help="Choose model type: ml (classic ML classifiers) or DL models like cnn"
    )
    parser.add_argument("--save_onnx", action="store_true",
                        help="If set, export trained sklearn models to ONNX")

    
    return parser.parse_args()


def main():
    args = parse_args()
    
    hand_crafted_features = True

    if args.dataset != "WISDM":
        raise ValueError(f"Dataset {args.dataset} not supported.")
    
    # Announce device before training starts
    print(f"Using device: {args.device}")

    # Collate/feature extraction setup (datasets will be created per fold below)
    if args.feature_extraction == "handcrafted":
        feature_extractor = HandCraftedFeaturesExtractor(device=args.device)

        def collate_with_features(batch):
            X, y = zip(*batch)
            X = torch.tensor(X, dtype=torch.float32, device=args.device)  # (B,200,3)
            feats = feature_extractor(X)  # (B,43)

            if args.model == "ml":
                # Convert to CPU numpy for sklearn
                return feats.cpu().numpy(), torch.tensor(y, dtype=torch.long).cpu().numpy()
            else:
                # Keep tensors on device for DL
                return feats, torch.tensor(y, dtype=torch.long, device=args.device)

        collate_fn = collate_with_features
    elif args.feature_extraction == "raw":
        # Raw features: flatten (B,200,3) -> (B,600)
        def collate_raw_concat(batch):
            X, y = zip(*batch)
            X = torch.tensor(X, dtype=torch.float32, device=args.device)  # (B,200,3)
            X = X.view(X.shape[0], -1)  # (B,600)

            if args.model == "ml":
                return X.cpu().numpy(), torch.tensor(y, dtype=torch.long).cpu().numpy()
            else:
                return X, torch.tensor(y, dtype=torch.long, device=args.device)

        collate_fn = collate_raw_concat

    if args.model == "ml":
        # Prepare ONNX exporter once (export only on first fold)
        onnx_ready = False
        if args.save_onnx:
            try:
                from skl2onnx import convert_sklearn
                from skl2onnx.common.data_types import FloatTensorType
                onnx_ready = True
            except Exception as e:
                print("[WARN] ONNX export requested but skl2onnx is not available.")
                print("       Install with: pip install skl2onnx onnx onnxruntime")
                onnx_ready = False
            if onnx_ready:
                os.makedirs("onnx_models", exist_ok=True)

        # Collect all train/test data (since sklearn needs arrays, not batches)
        # Define classifiers
        classifiers = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "KNN": KNeighborsClassifier(n_neighbors=5),
            "SVM": SVC(kernel="rbf"),
            "Random Forest": RandomForestClassifier(n_estimators=100),
            "Naive Bayes": GaussianNB(),
            "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                    solver="adam", max_iter=200, random_state=42),
        }
        # Metrics storage per classifier across folds
        metrics = {
            name: {"acc": [], "precision": [], "recall": [], "f1": [], "time_ms": []}
            for name in classifiers.keys()
        }

        for fold_idx in range(args.n_splits):
            if args.dataset != "WISDM":
                raise ValueError(f"Dataset {args.dataset} not supported.")

            # Build datasets per fold
            train_dataset = WISDMDataset(
                args.dataset_path,
                window_size_sec=5,
                stride_sec=1,
                k_fold=fold_idx,
                n_splits=args.n_splits,
                train=True,
            )
            test_dataset = WISDMDataset(
                args.dataset_path,
                window_size_sec=5,
                stride_sec=1,
                k_fold=fold_idx,
                n_splits=args.n_splits,
                train=False,
            )

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate_fn)

            # Materialize arrays for sklearn
            X_train, y_train = [], []
            X_test, y_test = [], []
            for Xb, yb in train_loader:
                X_train.append(Xb)
                y_train.append(yb)
            for Xb, yb in test_loader:
                X_test.append(Xb)
                y_test.append(yb)
            X_train = np.vstack(X_train)
            y_train = np.hstack(y_train)
            X_test = np.vstack(X_test)
            y_test = np.hstack(y_test)

            # Train & evaluate each classifier on this fold
            for name, base_clf in classifiers.items():
                clf = clone(base_clf)
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

                acc = accuracy_score(y_test, preds)
                precision = precision_score(y_test, preds, average="weighted", zero_division=0)
                recall = recall_score(y_test, preds, average="weighted", zero_division=0)
                f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

                # Inference time measurement (per sample)
                n_runs = len(X_test)
                start = time.perf_counter()
                _ = clf.predict(X_test)  # batch predict
                end = time.perf_counter()
                avg_time = (end - start) / max(n_runs, 1) * 1000.0  # ms/sample

                metrics[name]["acc"].append(acc)
                metrics[name]["precision"].append(precision)
                metrics[name]["recall"].append(recall)
                metrics[name]["f1"].append(f1)
                metrics[name]["time_ms"].append(avg_time)

                # Save ONNX only on first fold
                if fold_idx == 0 and args.save_onnx and onnx_ready:
                    try:
                        n_features = X_train.shape[1]
                        initial_types = [("input", FloatTensorType([None, n_features]))]
                        onnx_model = convert_sklearn(clf, initial_types=initial_types)
                        file_name = name.lower().replace(" ", "_") + ".onnx"
                        out_path = os.path.join("onnx_models", file_name)
                        with open(out_path, "wb") as f:
                            f.write(onnx_model.SerializeToString())
                        print(f"Saved ONNX: {out_path}")
                    except Exception as e:
                        print(f"[WARN] Failed to export {name} to ONNX: {e}")

        # After all folds, report mean ± std for each classifier
        print("\n" + "="*50)
        print("Aggregated ML performance over folds (mean ± std):")
        for name, m in metrics.items():
            acc_mu, acc_sd = np.mean(m["acc"]), np.std(m["acc"]) 
            pre_mu, pre_sd = np.mean(m["precision"]), np.std(m["precision"]) 
            rec_mu, rec_sd = np.mean(m["recall"]), np.std(m["recall"]) 
            f1_mu, f1_sd = np.mean(m["f1"]), np.std(m["f1"]) 
            t_mu, t_sd = np.mean(m["time_ms"]), np.std(m["time_ms"]) 
            print("- {}".format(name))
            print("  Accuracy:  {:.4f} ± {:.4f}".format(acc_mu, acc_sd))
            print("  Precision: {:.4f} ± {:.4f}".format(pre_mu, pre_sd))
            print("  Recall:    {:.4f} ± {:.4f}".format(rec_mu, rec_sd))
            print("  F1-score:  {:.4f} ± {:.4f}".format(f1_mu, f1_sd))
            print("  Inference: {:.4f} ± {:.4f} ms/sample".format(t_mu, t_sd))

    elif args.model == "cnn":
        # Repeated training/evaluation across folds
        accs, pres, recs, f1s, times_ms = [], [], [], [], []

        for fold_idx in range(args.n_splits):
            # Build datasets per fold
            train_dataset = WISDMDataset(
                args.dataset_path,
                window_size_sec=5,
                stride_sec=1,
                k_fold=fold_idx,
                n_splits=args.n_splits,
                train=True,
            )
            test_dataset = WISDMDataset(
                args.dataset_path,
                window_size_sec=5,
                stride_sec=1,
                k_fold=fold_idx,
                n_splits=args.n_splits,
                train=False,
            )

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                      shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, collate_fn=collate_fn)

            # Fresh model per fold
            model = SimpleCNN1D(num_classes=4)
            trainer = Trainer(model, device=args.device, lr=1e-3, epochs=10)
            trainer.fit(train_loader)

            # Evaluate with detailed metrics
            y_true, y_pred = [], []
            start = time.perf_counter()
            with torch.no_grad():
                trainer.model.eval()
                for Xb, yb in test_loader:
                    logits = trainer.model(Xb)
                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1).detach().cpu().numpy()
                    y_pred.append(preds)
                    y_true.append(yb.detach().cpu().numpy())
            end = time.perf_counter()
            y_true = np.hstack(y_true)
            y_pred = np.hstack(y_pred)
            n_samples = len(y_true)
            avg_time = (end - start) / max(n_samples, 1) * 1000.0

            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            print(acc, precision, recall, f1)

            accs.append(acc)
            pres.append(precision)
            recs.append(recall)
            f1s.append(f1)
            times_ms.append(avg_time)

            # ONNX export only on first fold
            if fold_idx == 0 and args.save_onnx:
                feat_len = None
                for Xb, _ in test_loader:
                    feat_len = int(Xb.shape[1])
                    break
                if feat_len is None:
                    for Xb, _ in train_loader:
                        feat_len = int(Xb.shape[1])
                        break
                if feat_len is not None:
                    out_path = os.path.join("onnx_models", "cnn.onnx")
                    trainer.export_onnx(sample_length=feat_len, out_path=out_path)
                else:
                    print("[WARN] Skipping ONNX export: could not infer feature length.")

        # Report aggregate metrics
        print("\n" + "="*50)
        print("Aggregated CNN performance over folds (mean ± std):")
        print("- SimpleCNN1D")
        print("  Accuracy:  {:.4f} ± {:.4f}".format(np.mean(accs), np.std(accs)))
        print("  Precision: {:.4f} ± {:.4f}".format(np.mean(pres), np.std(pres)))
        print("  Recall:    {:.4f} ± {:.4f}".format(np.mean(recs), np.std(recs)))
        print("  F1-score:  {:.4f} ± {:.4f}".format(np.mean(f1s), np.std(f1s)))
        print("  Inference: {:.4f} ± {:.4f} ms/sample".format(np.mean(times_ms), np.std(times_ms)))

if __name__ == "__main__":
    main()
