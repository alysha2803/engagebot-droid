"""
Phase 8: generates a ground-truth labeling template from a predictions CSV
(from run_on_video.py), so you don't have to guess timestamps or retype
student names by hand.

Samples every INTERVAL_SECONDS for each student that actually appears in
the predictions, with an empty true_level column for you to fill in while
watching the recording. Judge each sample point as you would describe the
student's actual engagement at that moment: "high", "medium", or "low" --
same three levels the system itself predicts, so the comparison in
evaluate.py is apples-to-apples. Leave a row blank and delete it if you
genuinely can't tell (occluded, off-screen, etc.) -- don't guess a label
you're not confident in, since a human's own doubtful dreamed-up "ground
truth" imposed as if it were the true label would quietly corrupt the whole
evaluation.

Usage:
    python make_gt_template.py predictions.csv --out ground_truth_template.csv --interval 10
"""
import argparse
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions_csv")
    parser.add_argument("--out", default="ground_truth_template.csv")
    parser.add_argument("--interval", type=float, default=10.0, help="Seconds between label points")
    args = parser.parse_args()

    with open(args.predictions_csv, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No prediction rows found -- nothing to template.")
        return

    students = sorted({r["student_name"] for r in rows if r["student_name"]})
    max_t = max(float(r["elapsed_seconds"]) for r in rows)

    template_rows = []
    t = 0.0
    while t <= max_t:
        for name in students:
            template_rows.append({"elapsed_seconds": round(t, 1), "student_name": name, "true_level": ""})
        t += args.interval

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["elapsed_seconds", "student_name", "true_level"])
        writer.writeheader()
        writer.writerows(template_rows)

    print(f"Wrote {len(template_rows)} label points for {len(students)} student(s) to {args.out}")
    print("Open it, watch the recording at each elapsed_seconds, and fill in true_level as high/medium/low.")
    print("Delete any row you're not confident labeling -- don't guess.")


if __name__ == "__main__":
    main()
