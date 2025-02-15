import nltk
import jax
import pandas as pd


def process_truncate_captions(
    captions, key, max_length_words=None, use_sum1=False, df_sum_merged=None
):
    """Process and truncate captions"""
    captions = captions.numpy().tolist()
    captions = [c.decode("utf-8") for c in captions]

    def get_objects_phenomena(caption):
        first_part = caption.split(";")[0]
        match = df_sum_merged[df_sum_merged["objects_phenomena_x"] == first_part][
            "objects_phenomena_y"
        ]
        return (
            match.values[0] if not (match.empty or pd.isna(match.values[0])) else "None"
        )

    # Get sum1 summaries by matching
    if use_sum1 and df_sum_merged is not None:
        results = pd.Series(captions).apply(get_objects_phenomena)
        captions = list(results.values)

    if max_length_words is not None:
        captions = sample_and_pad(captions, key, max_length=max_length_words)

    return captions


def tokenize_captions(
    captions, tokenizer, max_length_tokens, max_length_words=None, key=None
):
    """Tokenizer if using custom CLIP model"""
    captions = process_truncate_captions(
        captions, key, max_length_words=max_length_words
    )
    captions = tokenizer(
        captions,
        padding="max_length",
        truncation=True,
        max_length=max_length_tokens,
        return_tensors="np",
    )
    return captions["input_ids"], captions["attention_mask"]


def sample_and_pad(captions, key, max_length=77):
    """Sample a random sequence of sentences up to max_length in length"""
    sampled_seqs = []

    for caption in captions:
        sentences = nltk.sent_tokenize(caption)

        # If the total words are already less than or equal to max_length, just append the caption
        if len(caption.split()) <= max_length:
            sampled_seqs.append(caption)
            continue

        # Get random start index for sentences
        key, subkey = jax.random.split(key)
        start_idx = jax.random.randint(
            subkey, minval=0, maxval=len(sentences), shape=()
        )

        # Rotate the sentences so that the starting sentence is at the beginning
        rotated_sentences = sentences[start_idx:] + sentences[:start_idx]

        sampled_words = []
        for sent in rotated_sentences:
            if len(sampled_words) + len(sent.split()) <= max_length:
                sampled_words.extend(sent.split())
            else:
                break

        sampled_seq = " ".join(sampled_words)
        sampled_seqs.append(sampled_seq)

    return sampled_seqs
