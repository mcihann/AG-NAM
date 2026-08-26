from agnam.data.synthetic import generate_synthetic


def main():
    print(
        f"{'Scenario':<10}"
        f"{'n':>10}"
        f"{'p':>8}"
        f"{'Positive':>12}"
        f"{'Interactions':>15}"
    )

    print("-" * 55)

    for scenario in ("S1", "S2", "S3", "S4"):
        dataset = generate_synthetic(
            scenario=scenario,
            seed=42,
        )

        print(
            f"{scenario:<10}"
            f"{len(dataset.X):>10}"
            f"{dataset.X.shape[1]:>8}"
            f"{dataset.y.mean():>12.4f}"
            f"{len(dataset.true_interactions):>15}"
        )

        print(
            "  Ground truth:",
            dataset.true_interactions,
        )

        if scenario == "S3":
            print(
                "  Proxy pairs:",
                dataset.metadata["proxy_pairs"],
            )


if __name__ == "__main__":
    main()