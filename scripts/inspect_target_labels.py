from __future__ import annotations

import openml
import pandas as pd


TASK_IDS = [
    49,       # tic-tac-toe
    14952,    # PhishingWebsites
    29,       # credit-approval
    167141,   # churn
    14965,    # bank-marketing
    7592,     # adult
    9957,     # qsar-biodeg
    9978,     # ozone-level-8hr
    43,       # spambase
    3904,     # jm1
]


def main():
    rows = []

    for task_id in TASK_IDS:
        print(f"Loading task {task_id} ...")

        task = openml.tasks.get_task(
            task_id,
            download_splits=False
        )

        dataset = openml.datasets.get_dataset(task.dataset_id)

        _, y, _, _ = dataset.get_data(
            dataset_format="dataframe",
            target=task.target_name
        )

        y = pd.Series(y)

        counts = y.value_counts(dropna=False)

        labels = [str(label) for label in counts.index.tolist()]
        count_values = counts.tolist()

        rows.append(
            {
                "task_id": task_id,
                "dataset_name": dataset.name,
                "target_name": task.target_name,
                "class_1": labels[0],
                "class_1_count": count_values[0],
                "class_2": labels[1],
                "class_2_count": count_values[1],
            }
        )

    result = pd.DataFrame(rows)

    print("\n========================================")
    print("TARGET LABEL AUDIT")
    print("========================================")

    print(result.to_string(index=False))

    result.to_csv(
        "docs/target_label_audit.csv",
        index=False
    )

    print("\nSaved:")
    print("docs/target_label_audit.csv")


if __name__ == "__main__":
    main()