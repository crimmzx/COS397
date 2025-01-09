from models import (
    PyCantoneseModel,
    ToJyutpingModel,
    CantoneseG2PWModel,
    G2PM_Model,
)
import time
from data import prepare_data, calculate_metrics
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--max-samples", type=int, default=None)
    args = args.parse_args()

    model_classes = [
        PyCantoneseModel,
        ToJyutpingModel,
        CantoneseG2PWModel,
        G2PM_Model,
    ]
    model_names = []

    wordshk_texts, wordshk_query_ids, wordshk_phonemes, _ = prepare_data(
        "data/wordshk.sent", "data/wordshk.lb", args.max_samples
    )
    colloquial_texts, colloquial_query_ids, colloquial_phonemes, colloquial_pos = (
        prepare_data(
            "data/colloquial.sent", "data/colloquial.lb", "data/colloquial.pos"
        )
    )
    classical_texts, classical_query_ids, classical_phonemes, _ = prepare_data(
        "data/classical.sent", "data/classical.lb"
    )

    test_texts = wordshk_texts + colloquial_texts + classical_texts
    test_query_ids = wordshk_query_ids + colloquial_query_ids + classical_query_ids
    test_phonemes = wordshk_phonemes + colloquial_phonemes + classical_phonemes
    test_pos = colloquial_pos

    results = {}

    for model_class in model_classes:
        model = model_class()
        model_name = model.get_name()

        print(f"\nTesting {model_name}...")
        start_time = time.time()
        predictions = []
        for text in tqdm(test_texts, desc=f"Predicting with {model_name}"):
            predictions.append(model([text])[0])
        runtime = time.time() - start_time

        acc, per, precision, recall, f1, pos_acc = calculate_metrics(
            predictions, test_texts, test_query_ids, test_phonemes, test_pos
        )

        print(f"Accuracy: {acc:.4f}")
        print(f"Phoneme Error Rate (PER): {per:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if pos_acc:
            for pos, pos_acc_value in pos_acc.items():
                print(f"POS Tag '{pos}' Accuracy: {pos_acc_value:.4f}")
        print(f"Runtime: {runtime:.4f}s")

        model_names.append(model_name)
        results[model_name] = {
            "accuracy": acc,
            "per": per,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pos_accuracy": pos_acc,
            "runtime": runtime,
        }

    colors = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
    ]
    width = 0.2

    plt.figure(figsize=(14, 8))
    x = range(len(model_names))
    accuracy_bars = plt.bar(
        [i - 0.5 * width for i in x],
        [results[model_name]["accuracy"] for model_name in model_names],
        width,
        label="Accuracy",
        color=colors[0],
    )
    per_bars = plt.bar(
        [i + 0.5 * width for i in x],
        [results[model_name]["per"] for model_name in model_names],
        width,
        label="Phoneme Error Rate (PER)",
        color=colors[1],
    )
    plt.xticks(x, model_names)

    plt.ylabel("Scores")
    plt.title("G2P Model Performance")
    plt.legend(loc="upper right")

    for bar in accuracy_bars + per_bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
        )

    plt.savefig("result.png")

    plt.figure(figsize=(14, 8))
    runtime_bars = plt.bar(
        model_names,
        [results[model_name]["runtime"] for model_name in model_names],
        width,
        label="Runtime",
        color=colors[5],
    )

    plt.ylabel("Runtime (s)")
    plt.title("G2P Model Runtime")
    plt.legend(loc="upper right")

    # Add text labels on the bars
    for bar in runtime_bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
        )

    plt.savefig("runtime.png")

    plt.figure(figsize=(14, 8))
    precision_bars = plt.bar(
        x,
        [results[model_name]["precision"] for model_name in model_names],
        width / 4,
        label="Precision",
        color="#e41a1c",
    )
    recall_bars = plt.bar(
        [i + width for i in x],
        [results[model_name]["recall"] for model_name in model_names],
        width / 4,
        label="Recall",
        color="#377eb8",
    )
    f1_bars = plt.bar(
        [i + 2 * width for i in x],
        [results[model_name]["f1"] for model_name in model_names],
        width / 4,
        label="F1-Score",
        color="#4daf4a",
    )
    plt.xticks([i + width for i in x], model_names)

    plt.ylabel("Scores")
    plt.title("G2P Model Precision, Recall, and F1-Score")
    plt.legend(loc="upper right")

    for bar in precision_bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
        )
    for bar in recall_bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
        )
    for bar in f1_bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.4f}",
            ha="center",
            va="bottom",
        )

    plt.savefig("precision_recall_f1.png")

    if test_pos:
        pos_tags = list(set(test_pos))
        pos_labels = {
            "v": "Verb",
            "i": "Interjection",
            "p": "Preposition",
            "t": "Particle",
            "d": "Determiner",
            "unk": "Unknown",
            "n": "Noun",
            "c": "Conjunction",
        }
        for model_name in model_names:
            plt.figure(figsize=(14, 8))
            pos_bars = plt.bar(
                pos_tags,
                [results[model_name]["pos_accuracy"].get(pos, 0) for pos in pos_tags],
                width,
                label=f"POS Tag Accuracy for {model_name}",
                color=colors[4],
            )
            plt.xticks(pos_tags)

            plt.ylabel("Accuracy")
            plt.title(f"G2P Model POS Tag Accuracy for {model_name}")
            plt.legend(loc="lower left")

            for bar in pos_bars:
                yval = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    yval + 0.01,
                    f"{yval:.4f}",
                    ha="center",
                    va="bottom",
                )

            handles, labels = plt.gca().get_legend_handles_labels()
            for pos, label in pos_labels.items():
                handles.append(plt.Rectangle((0, 0), 1, 1, color="white", ec="black"))
                labels.append(f"{pos} ({label})")
            plt.legend(handles, labels, loc="lower left")

            plt.savefig(f"pos_accuracy_{model_name}.png")