import re


def clean_sentence(text):
    text = text.encode("utf-8").decode("utf-8")
    text = re.compile(r"http\S+").sub("", text)
    text = re.compile(r"<.*?>").sub("", text)
    whitelist = set(
        "abcdefghijklmnopqrstuvwxyz ÄÖÜäöüß ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}:;-&$@#%£€/\\|_+*¥"
    )
    text = "".join(char.lower() for char in text if char in whitelist)
    return text


def filter_by_length(source, target, min_length, max_length):
    return (
        min_length <= len(source.split()) <= max_length
        and min_length <= len(target.split()) <= max_length
    )


def preprocess(example, min_length=5, max_length=64, ratio=1.5):
    source = clean_sentence(example["de"])
    target = clean_sentence(example["en"])
    if filter_by_length(source, target, min_length, max_length):
        example["src"] = source
        example["tgt"] = target
        return example
    else:
        return None


def clean_dataset(dataset):

    cleaned_dataset = []

    for example in dataset["translation"]:
        cleaned_example = preprocess(example)
        if cleaned_example is None:
            continue
        cleaned_dataset.append(cleaned_example)

    return cleaned_dataset
