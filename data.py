import re
from functools import reduce
from models import G2PModel
from sklearn.metrics import precision_score, recall_score, f1_score

# The choice of symbols is arbitrary and does not affect the calculation and the final result.
# As long as each symbol represents a unique phoneme in its position,
# it is fine even if the same symbol represents different phonemes in different positions.
JYUTPING_TO_PHONEME_RULES = {
    r"ng": "N",
    r"g(w|(?=u(?!N|k)))": "G",  # replace onsets of syllables beginning with gw- or gu-, excluding gung and guk
    r"k(w|(?=u(?!N|k)))": "K",  # replace onsets of syllables beginning with kw- or ku-, excluding kung and kuk
    r"aa": "A",
    r"oe": "O",
    r"eo": "E",
    r"yu": "Y",
    r"e(?=i)|i(?=N|k)": "I",    # replace nuclei of syllables ending with -ei, -ing or -ik
    r"o(?=u)|u(?=N|k)": "U",    # replace nuclei of syllables ending with -ou, -ung or -uk
    r"^(?=[mN]\d)": "__",       # add null onsets and nuclei to syllabic m and ng
    r"(?<=^h)(?=[mN]\d)": "_",  # add null nuclei to syllabic hm and hng
    r"^(?=[aeiouAEIOUY])": "_", # add null onsets
    r"(?<=^..)(?=\d)": "_",     # add null codas
}

PHONEMES_PER_SYLLABLE = 4

ANCHOR_CHAR = "▁"


def jyutping_to_phonemes(jyutping):
    phonemes = reduce(lambda pron, rule: re.sub(*rule, pron), JYUTPING_TO_PHONEME_RULES.items(), jyutping)
    if len(phonemes) != PHONEMES_PER_SYLLABLE:
        # phonemes must be in the format (onset, nucleus, coda, tone)
        raise ValueError(f"Invalid jyutping: {jyutping}, {phonemes}")
    return phonemes


def prepare_data(sent_path, lb_path=None, pos_path=None, max_samples=None):
    raw_texts = open(sent_path, encoding="utf-8").read().rstrip().split("\n")
    query_ids = [raw.index(ANCHOR_CHAR) for raw in raw_texts]
    texts = [raw.replace(ANCHOR_CHAR, "") for raw in raw_texts]
    if lb_path is None:
        if max_samples:
            return texts[:max_samples], query_ids[:max_samples], None, None

        return texts, query_ids, None, None
    else:
        jyutpings = open(lb_path, encoding="utf-8").read().rstrip().split("\n")
        pos_tags = (
            open(pos_path, encoding="utf-8").read().rstrip().split("\n")
            if pos_path
            else None
        )

        if max_samples:
            texts = texts[:max_samples]
            query_ids = query_ids[:max_samples]
            jyutpings = jyutpings[:max_samples]
            pos_tags = pos_tags[:max_samples] if pos_tags else None

        phonemes = [list({jyutping_to_phonemes(jyutping) for jyutping in alternatives.split("/")}) for alternatives in jyutpings]

        return texts, query_ids, phonemes, pos_tags

def calculate_metrics(
    predictions, test_texts, test_query_ids, ground_truths, pos_tags=None
):
    """
    Calculates the accuracy, Phoneme Error Rate (PER), precision, recall, F1 score, and POS accuracy for monosyllabic predictions.

    Args:
      predictions: A list of predicted phonemes.
      ground_truths: A list of ground truth phonemes.
      pos_tags: A list of part-of-speech tags.

    Returns:
      (Accuracy score, PER score, Precision score, Recall score, F1 score, POS accuracy)
    """
    total_predictions = 0
    correct_predictions = 0
    total_errors = 0
    all_true_phonemes = []
    all_pred_phonemes = []
    pos_dict = {}

    for text, query_id, true_phonemes, pred_jyutpings, pos in zip(
        test_texts, test_query_ids, ground_truths, predictions, pos_tags
    ):
        if len(pred_jyutpings) > query_id:
            total_predictions += 1
            predicted_jyutping = pred_jyutpings[query_id]
            if predicted_jyutping is None:
                total_errors += PHONEMES_PER_SYLLABLE
                all_pred_phonemes.append(None)
                all_true_phonemes.append(list(true_phonemes)[0])
            else:
                predicted_phonemes = jyutping_to_phonemes(predicted_jyutping)
                all_pred_phonemes.append(predicted_phonemes)
                all_true_phonemes.append(list(true_phonemes)[0])
                if predicted_phonemes in true_phonemes:
                    correct_predictions += 1
                    if pos not in pos_dict:
                        pos_dict[pos] = {"correct": 0, "total": 0}
                    pos_dict[pos]["correct"] += 1
                min_distance = PHONEMES_PER_SYLLABLE
                for alternative in true_phonemes:
                    distance = 0
                    for i in range(PHONEMES_PER_SYLLABLE):
                        if predicted_phonemes[i] != alternative[i]:
                            distance += 1
                    if distance < min_distance:
                        min_distance = distance
                total_errors += min_distance
                if pos not in pos_dict:
                    pos_dict[pos] = {"correct": 0, "total": 0}
                pos_dict[pos]["total"] += 1

    filtered_true_phonemes, filtered_pred_phonemes = zip(*[
        (true, pred) for true, pred in zip(all_true_phonemes, all_pred_phonemes) if true is not None and pred is not None
    ])

    acc = correct_predictions / total_predictions
    per = total_errors / total_predictions / PHONEMES_PER_SYLLABLE
    precision = precision_score(
        filtered_true_phonemes, filtered_pred_phonemes, average="macro", zero_division=0
    )
    recall = recall_score(
        filtered_true_phonemes, filtered_pred_phonemes, average="macro", zero_division=0
    )
    f1 = f1_score(
        filtered_true_phonemes, filtered_pred_phonemes, average="macro", zero_division=0
    )

    pos_acc = {pos: data["correct"] / data["total"] for pos, data in pos_dict.items()}

    return acc, per, precision, recall, f1, pos_acc


def test(
    model: G2PModel, sent_path="data/test.sent", lb_path="data/test.lb", pos_path=None
):
    test_texts, test_query_ids, test_phonemes, test_pos = prepare_data(
        sent_path, lb_path, pos_path
    )
    predictions = model(test_texts)

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