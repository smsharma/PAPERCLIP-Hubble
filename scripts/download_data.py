import os
import numpy as np
from astroquery.mast import Observations
from PIL import Image


def downsample_image(image_path, max_size):
    """
    Downsample an image to have a maximum side length of max_size.

    Parameters:
    - image_path: Path to the image to be downsampled.
    - max_size: The maximum side length for the output image.
    """
    with Image.open(image_path) as img:
        img_ratio = img.width / img.height
        if img.width > img.height:
            new_width = max_size
            new_height = int(max_size / img_ratio)
        else:
            new_height = max_size
            new_width = int(max_size * img_ratio)

        img = img.resize((new_width, new_height), Image.LANCZOS)
        img.save(image_path)


def download_data(proposal_id, n_max_images, max_resolution, seed, data_dir="../data", exclude_color=False):
    download_dir = f"{data_dir}/proposal_{proposal_id}"

    # Query
    obs_table = Observations.query_criteria(
        obs_collection="HST",
        proposal_id=[f"{proposal_id}"],
        dataRights="PUBLIC",
    )

    # If no observations found, return
    if len(obs_table) == 0:
        print(f"No observations found for proposal {proposal_id}")
        return

    # Get preview products
    products = Observations.get_product_list(obs_table)
    products = products[products["productType"] == "PREVIEW"]

    # Only total preview image
    match = "total"

    # Potentially xclude images with "color" in the filename
    if exclude_color:
        exclude = "color"
    else:
        exclude = "dont_exclude_anything"  # Dummy

    match_mask = [match in row["productFilename"] for row in products]
    exclude_mask = [exclude not in row["productFilename"] for row in products]

    mask = [m and e for m, e in zip(match_mask, exclude_mask)]

    # Check if there are any images
    if not np.any(mask):
        print(f"No images found for proposal {proposal_id}")
        return

    # Randomly select up to 20 images
    products = products[mask][np.random.RandomState(seed=seed).choice(np.arange(len(products[mask])), n_max_images)]

    # Download
    Observations.download_products(
        products,
        extension=["jpg", "jpeg"],
        productType="PREVIEW",
        download_dir=download_dir,
    )

    # Recursively find all images in "total_images" directory and bring them up to "total_images"
    for root, dirs, files in os.walk(download_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg")):
                source = os.path.join(root, file)
                destination = os.path.join(download_dir, file)

                # Check if file already exists in the destination
                if not os.path.exists(destination):
                    os.rename(source, destination)

    # Remove the now-empty subdirectories:
    for dirpath, dirnames, filenames in os.walk(download_dir, topdown=False):
        for dirname in dirnames:
            dir_to_remove = os.path.join(dirpath, dirname)
            if not os.listdir(dir_to_remove):  # directory is empty
                os.rmdir(dir_to_remove)

    # After you bring the images up to "total_images", downsample them:
    for root, dirs, files in os.walk(download_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                # Try downsampling the image
                try:
                    downsample_image(image_path, max_resolution)
                # Otherwise, delete it
                except:
                    os.remove(image_path)
