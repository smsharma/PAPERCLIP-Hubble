import tensorflow as tf
import jax
import flax


# Parse function
def parse_function(example):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "abstract": tf.io.FixedLenFeature([], tf.string),
        "image_height": tf.io.FixedLenFeature([], tf.int64),
        "image_width": tf.io.FixedLenFeature([], tf.int64),
    }

    parsed_features = tf.io.parse_single_example(example, features)

    image = parsed_features["image"]
    caption = parsed_features["abstract"]
    image_height = parsed_features["image_height"]
    image_width = parsed_features["image_width"]

    # Decode raw image bytes
    image = tf.io.decode_raw(image, tf.uint8)
    image = tf.reshape(image, [image_height, image_width, 3])

    return image, caption


def _normalize(image, abstract):
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, abstract


def make_dataloader(files, batch_size, seed, train_fraction=0.8, split="train", shuffle=True):
    ds = tf.data.TFRecordDataset(files)

    # Count total examples 
    num_total = sum(1 for _ in ds)
    num_train = int(train_fraction * num_total)

    if split == 'train':
        ds = ds.take(num_train)
    elif split == 'val':
        ds = ds.skip(num_train)
    else:
        raise ValueError(f"Invalid split {split}")

    ds = ds.map(parse_function)
    ds = ds.map(_normalize)
    ds = ds.cache()

    if split == 'train':
        ds = ds.repeat()
    elif split == 'val':
        ds = ds.repeat(1)
        
    # Shuffle and batch
    if shuffle:
        ds = ds.shuffle(10000, seed=seed)
    ds = ds.batch(batch_size, drop_remainder=False)

    return ds


def create_input_iter(ds):
    """Create an input iterator that prefetches to device."""

    def _prepare(xs):
        def _f(x):
            x = x._numpy()
            return x

        return jax.tree_util.tree_map(_f, xs)

    it = map(_prepare, ds)
    it = flax.jax_utils.prefetch_to_device(it, 2)
    return it
