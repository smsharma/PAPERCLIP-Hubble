import sys, os
sys.path.append("../")

import outlines.models as models
import outlines.text as text

import argparse
from tqdm import tqdm
import pandas as pd

from utils.summarize_utils import prompt_fn, ConstrainedResponseHST
from utils.abstract_utils import read_abstracts_file

def download(
    data_dir: str = "./data/",
    observations_dir: str = "observations_v1",
    n_max_abstracts: int = 9999999,
    abstract_filename: str = "abstracts.cat",
    model_name_awq: str = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ",
    save_filename: str = "summary_v1",
    seed: int = 42,
    verbose: bool = False,
    n_max_tries: int = 5,
):
    
    # Load the model
    model = models.awq(model_name_awq)

    # Load the abstracts
    abstract_file = f"{data_dir}/{abstract_filename}"
    abstracts_df = read_abstracts_file(abstract_file)

    # Drop rows with missing Cycle
    abstracts_df = abstracts_df.dropna(subset=['Cycle'])
    abstracts_df = abstracts_df[abstracts_df['Cycle'] != '']

    # Convert Cycle and ID to int
    abstracts_df['Cycle'] = abstracts_df['Cycle'].astype(int)
    abstracts_df['ID'] = abstracts_df['ID'].astype(int)

    # Lists to store results
    proposal_id_list = []
    objects_list = []
    science_list = []

    # Collect directories that contain .jpg files and match the "proposal_" pattern, excluding unwanted directories
    directories_with_images = [os.path.join(r, d)
                               for r, dirs, files in os.walk(f"{data_dir}/{observations_dir}/")
                               for d in dirs
                               if d.startswith("proposal_") and not d.endswith('.ipynb_checkpoints')]

    # Walk through data folder
    for directory in tqdm(directories_with_images[:n_max_abstracts]):
        proposal_id = directory.split("proposal_")[-1]  # Extract proposal id from the directory name

        # Extract abstract using the dataframe
        abstract = abstracts_df[abstracts_df["ID"] == int(proposal_id)]["Abstract"].values[0]

        # Generate summary with constrained response
        prompt = prompt_fn(abstract)
        generator = text.generate.json(model, ConstrainedResponseHST)

        # Try up to n_max_tries times; if it fails, skip and go to the next abstract
        for _ in range(n_max_tries):
            try:
                result = generator(prompt)
                break
            except:
                pass

        if verbose:
            print(result)
            print("\n")

        proposal_id_list.append(proposal_id)
        science_list.append(result.science_use_cases)
        objects_list.append(result.objects_and_phenomena)

    # Create a DataFrame
    df = pd.DataFrame({
        'proposal_id': proposal_id_list,
        'objects_phenomena': objects_list,
        'science_use_cases': science_list
    })

    # Save the DataFrame
    df.to_csv(f"{data_dir}/{save_filename}.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize abstracts.")
    parser.add_argument("--data_dir", type=str, default="./data/", help="Directory to save the downloaded data.")
    parser.add_argument("--observations_dir", type=str, default="observations_v2", help="Observations dir within the `data_dir`.")
    parser.add_argument("--n_max_abstracts", type=int, default=9999999, help="Maximum number of abstracts to summarize.")
    parser.add_argument("--abstract_filename", type=str, default="abstracts.cat", help="Filename of the abstracts file.")
    parser.add_argument("--model_name_awq", type=str, default="TheBloke/OpenHermes-2.5-Mistral-7B-AWQ", help="Model name for AWQ.")
    parser.add_argument("--save_filename", type=str, default="summary_v1", help="Filename to save the summary to.")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--n_max_tries", type=int, default=5, help="Maximum number of tries to generate a summary.")

    args = parser.parse_args()

    download(
        data_dir=args.data_dir,
        observations_dir=args.observations_dir,
        n_max_abstracts=args.n_max_abstracts,
        abstract_filename=args.abstract_filename,
        model_name_awq=args.model_name_awq,
        save_filename=args.save_filename,
        seed=args.seed,
        verbose=args.verbose,
        n_max_tries=args.n_max_tries,
    )
