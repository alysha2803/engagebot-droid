"""
Phase 8: compares run_on_video.py's predictions against human-labeled
ground truth and reports precision/recall/F1/confusion matrix for the
engagement level classification (high/medium/low) -- the project's own
NFR02 metric.

Matching: for each ground-truth row (elapsed_seconds, student_name,
true_level), finds that student's nearest prediction in time, within
--tolerance seconds. A ground-truth row with no prediction close enough is
reported as UNMATCHED rather than silently dropped or guessed -- a high
unmatched count usually means recognition lost that student around that
time, which is itself a finding, not just missing data.

No external ML library required (no scikit-learn dependency) -- precision/
recall/F1 for 3 classes is simple enough to compute directly and keeps this
script runnable with nothing beyond the stdlib.

Usage:
    python evaluate.py predictions.csv ground_truth.csv --tolerance 5
"""
import argparse
import csv
from collections import defaultdict

LEVELS = ["high", "medium", "low"]


def load_predictions(path):
    by_student = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_student[row["student_name"]].append(
                (float(row["elapsed_seconds"]), row["level"])
            )
    for name in by_student:
        by_student[name].sort()
    return by_student


def nearest_prediction(preds_for_student, t, tolerance):
    best = None
    best_dt = None
    for pt, level in preds_for_student:
        dt = abs(pt - t)
        if dt <= tolerance and (best_dt is None or dt < best_dt):
            best, best_dt = level, dt
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv")
    parser.add_argument("ground_truth_csv")
    parser.add_argument("--tolerance", type=float, default=5.0, help="Max seconds between a ground-truth point and its matched prediction")
    args = parser.parse_args()

    predictions = load_predictions(args.predictions_csv)

    pairs = []       # (true_level, pred_level)
    unmatched = []    # ground-truth rows with no nearby prediction

    with open(args.ground_truth_csv, newline="") as f:
        for row in csv.DictReader(f):
            true_level = row["true_level"].strip().lower()
            if not true_level:
                continue   # left blank on purpose -- not confident, skip
            if true_level not in LEVELS:
                print(f"  [WARN] skipping row with unrecognized true_level={true_level!r}")
                continue

            name = row["student_name"]
            t = float(row["elapsed_seconds"])
            pred = nearest_prediction(predictions.get(name, []), t, args.tolerance)
            if pred is None:
                unmatched.append((t, name))
            else:
                pairs.append((true_level, pred))

    if not pairs:
        print("No matched ground-truth rows -- nothing to evaluate. Check student names match exactly between the two CSVs.")
        return

    # Confusion matrix: confusion[true][pred] = count
    confusion = {t: {p: 0 for p in LEVELS} for t in LEVELS}
    for true_level, pred_level in pairs:
        confusion[true_level][pred_level] += 1

    total = len(pairs)
    correct = sum(confusion[l][l] for l in LEVELS)
    accuracy = correct / total

    print(f"\nMatched {total} ground-truth points ({len(unmatched)} unmatched, tolerance={args.tolerance}s)")
    print(f"Overall accuracy: {accuracy:.1%}\n")

    print("Confusion matrix (rows=true, cols=predicted):")
    header = "            " + "".join(f"{p:>10}" for p in LEVELS)
    print(header)
    for t in LEVELS:
        print(f"{t:>10}  " + "".join(f"{confusion[t][p]:>10}" for p in LEVELS))

    print("\nPer-class precision / recall / F1:")
    f1s = []
    for level in LEVELS:
        tp = confusion[level][level]
        fp = sum(confusion[t][level] for t in LEVELS if t != level)
        fn = sum(confusion[level][p] for p in LEVELS if p != level)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        f1s.append(f1)
        print(f"  {level:>8}:  precision={precision:.2f}  recall={recall:.2f}  f1={f1:.2f}")

    macro_f1 = sum(f1s) / len(f1s)
    print(f"\nMacro-averaged F1: {macro_f1:.2f}")

    if unmatched:
        print(f"\n{len(unmatched)} ground-truth point(s) had no prediction within {args.tolerance}s -- "
              "likely a recognition/tracking gap at that moment, worth checking:")
        for t, name in unmatched[:10]:
            print(f"  t={t:.1f}s  student={name}")
        if len(unmatched) > 10:
            print(f"  ... and {len(unmatched) - 10} more")


if __name__ == "__main__":
    main()
