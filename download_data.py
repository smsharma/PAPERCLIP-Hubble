import argparse
from tqdm import tqdm
from scripts.abstract_utils import read_abstracts_file
from scripts.download_data import download_data
from absl import logging


def download(
    max_resolution: int = 512,
    cycle_min: int = 25,
    cycle_max: int = 26,
    n_max_images: int = 10,
    seed: int = 42,
    data_dir: str = "./data/",
    observations_dir: str = "observations_v1",
    filename: str = "abstracts.cat",
    exclude_color: bool = False,
):
    abstracts_df = read_abstracts_file(f"{data_dir}/{filename}")

    # Drop rows with missing Cycle
    abstracts_df = abstracts_df.dropna(subset=["Cycle"])
    abstracts_df = abstracts_df[abstracts_df["Cycle"] != ""]

    # Convert Cycle and ID to int
    abstracts_df["Cycle"] = abstracts_df["Cycle"].astype(int)
    abstracts_df["ID"] = abstracts_df["ID"].astype(int)

    # Only keep specific Cycles
    abstracts_cycle_df = abstracts_df[(abstracts_df["Cycle"] >= cycle_min) & (abstracts_df["Cycle"] <= cycle_max)]

    # Log how many abstracts
    logging.info(f"Number of abstracts: {len(abstracts_cycle_df)}")

    abstract_ids = abstracts_cycle_df["ID"].values

    # Log how many abstracts broken down by Cycle
    logging.info(f"Number of abstracts by Cycle: {abstracts_cycle_df.groupby('Cycle').size()}")

    for proposal_id in tqdm(abstract_ids):
        download_data(proposal_id=proposal_id, n_max_images=n_max_images, max_resolution=max_resolution, seed=seed, data_dir=f"{data_dir}/{observations_dir}/", exclude_color=exclude_color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Data Based on Command Line Arguments")
    parser.add_argument("--max_resolution", type=int, default=512, help="Maximum resolution for the downloaded images.")
    parser.add_argument("--cycle_min", type=int, default=25, help="Minimum cycle for filtering abstracts.")
    parser.add_argument("--cycle_max", type=int, default=26, help="Maximum cycle for filtering abstracts.")
    parser.add_argument("--n_max_images", type=int, default=10, help="Maximum number of images to download.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for any random operation.")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Directory to save the downloaded data.")
    parser.add_argument("--observations_dir", type=str, default="observations_v2", help="Observations dir within the `data_dir`.")
    parser.add_argument("--exclude_color", action="store_true", help="Exclude images with `color` in the filename.")

    args = parser.parse_args()

    download(
        max_resolution=args.max_resolution,
        cycle_min=args.cycle_min,
        cycle_max=args.cycle_max,
        n_max_images=args.n_max_images,
        seed=args.seed,
        data_dir=args.data_dir,
        observations_dir=args.observations_dir,
        exclude_color=args.exclude_color,
    )
