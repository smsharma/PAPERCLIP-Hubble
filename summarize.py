import sys, os

sys.path.append("../")

import outlines
import outlines.models as models
import outlines.text as text

import argparse
from functools import partial
from tqdm import tqdm
import pandas as pd

import transformers
from transformers import BitsAndBytesConfig
import torch
from utils.summarize_utils import prompt_fn, ConstrainedResponseHST
from utils.abstract_utils import read_abstracts_file


def summarize(
    data_dir: str = "./data/",
    observations_dir: str = "observations_v1",
    i_start: int = 0,
    n_max_abstracts: int = 9999999,
    abstract_filename: str = "abstracts.cat",
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",  # "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ",
    save_filename: str = "summary_v1",
    seed: int = 42,
    verbose: bool = False,
    n_max_tries: int = 5,
):
    if model_name == "mistralai/Mixtral-8x7B-Instruct-v0.1":
        config = transformers.AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            asd=True,
        )

        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)

        model = models.transformers(
            model_name=model_name,
            model_kwargs={"config": config, "quantization_config": bnb_config, "trust_remote_code": True, "device_map": "auto", "load_in_4bit": True, "cache_dir": "/n/holystore01/LABS/iaifi_lab/Users/smsharma/hf_cache/"},
        )

    elif model_name == "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ":
        model = models.awq(model_name)
    else:
        raise ValueError("Invalid model name.")

    # Load the abstracts
    abstract_file = f"{data_dir}/{abstract_filename}"
    abstracts_df = read_abstracts_file(abstract_file)

    # Drop rows with missing Cycle
    abstracts_df = abstracts_df.dropna(subset=["Cycle"])
    abstracts_df = abstracts_df[abstracts_df["Cycle"] != ""]

    # Convert Cycle and ID to int
    abstracts_df["Cycle"] = abstracts_df["Cycle"].astype(int)
    abstracts_df["ID"] = abstracts_df["ID"].astype(int)

    # Lists to store results
    proposal_id_list = []
    objects_list = []
    science_list = []

    # Collect directories that contain .jpg files and match the "proposal_" pattern, excluding unwanted directories
    directories_with_images = [os.path.join(r, d) for r, dirs, files in os.walk(f"{data_dir}/{observations_dir}/") for d in dirs if d.startswith("proposal_") and not d.endswith(".ipynb_checkpoints")]

    generator = outlines.generate.json(model, ConstrainedResponseHST)

    # Walk through data folder
    for directory in tqdm(directories_with_images[i_start : i_start + n_max_abstracts]):
        proposal_id = directory.split("proposal_")[-1]  # Extract proposal id from the directory name

        # Extract abstract using the dataframe
        abstract = abstracts_df[abstracts_df["ID"] == int(proposal_id)]["Abstract"].values[0]

        # Generate summary with constrained response
        prompt = prompt_fn(abstract)

        # Try up to n_max_tries times; if it fails, skip and go to the next abstract
        for _ in range(n_max_tries):
            result = None
            try:
                result = generator(prompt)
                break
            except:
                pass

        if result is None:
            raise ValueError(f"Failed to generate summary for proposal {proposal_id}.")

        if verbose:
            # Print the abstract
            print("\n")
            print("Abstract:")
            print(abstract)
            print("\n")

            # Print the summary
            print("Summary:")
            print(result)
            print("\n")

            print("Objects and phenomena:")
            for i, obj in enumerate(result.objects_and_phenomena.split(", ")):
                print(f"{i+1}. {obj}")
            print("\n")

            print("Science use cases:")
            for i, sci in enumerate(result.science_use_cases.split(", ")):
                print(f"{i+1}. {sci}")
            print("\n")

        proposal_id_list.append(proposal_id)
        science_list.append(result.science_use_cases)
        objects_list.append(result.objects_and_phenomena)

    # Create a DataFrame
    df = pd.DataFrame({"proposal_id": proposal_id_list, "objects_phenomena": objects_list, "science_use_cases": science_list})

    # Save the DataFrame
    df.to_csv(f"{data_dir}/{save_filename}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize abstracts.")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Directory to save the downloaded data.")
    parser.add_argument("--observations_dir", type=str, default="observations_v2", help="Observations dir within the `data_dir`.")
    parser.add_argument("--i_start", type=int, default=0, help="Index to start at.")
    parser.add_argument("--n_max_abstracts", type=int, default=9999999, help="Maximum number of abstracts to summarize.")
    parser.add_argument("--abstract_filename", type=str, default="abstracts.cat", help="Filename of the abstracts file.")
    parser.add_argument("--model_name", type=str, default="mistralai/Mixtral-8x7B-Instruct-v0.1", help="Model name.")
    parser.add_argument("--save_filename", type=str, default="summary_v1", help="Filename to save the summary to.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_max_tries", type=int, default=5, help="Maximum number of tries to generate a summary.")

    args = parser.parse_args()

    summarize(
        data_dir=args.data_dir,
        observations_dir=args.observations_dir,
        i_start=args.i_start,
        n_max_abstracts=args.n_max_abstracts,
        abstract_filename=args.abstract_filename,
        model_name=args.model_name,
        save_filename=args.save_filename,
        seed=args.seed,
        verbose=args.verbose,
        n_max_tries=args.n_max_tries,
    )
