import outlines
import outlines.models as models

from pydantic import BaseModel, Field, constr, conlist, validator
from enum import Enum

import re

class ConstrainedResponseHST(BaseModel):
    # objects_and_phenomena: conlist(str, min_length=1, max_length=5)
    # science_use_cases: conlist(str, min_length=1, max_length=5)
    objects_and_phenomena: str
    science_use_cases: str

    # @validator('*', pre=True, each_item=True)
    # def clean_and_validate_strings(cls, v):
    #     if isinstance(v, str):
    #         # Strip leading unwanted characters
    #         v = v.lstrip(':, ')

    #         # # Regular expression to allow alphanumeric, spaces, and specific symbols
    #         # allowed = re.compile(r"^[a-zA-Z0-9\s\.\-\+\(\)\*\/]*$")
    #         # if not allowed.match(v) or not v.strip():
    #         #     raise ValueError('Invalid string. Allowed: alphanumeric, spaces, ".", "-", "+", "(", ")", "*", "/"')

    #     return v

@outlines.prompt
def prompt_fn(abstract):
        
    """<s>[INST] You are an expert astrophysicist, with broad expertise across observational and theoretical astrophysics. You are able to extract core information from astrophysical texts.

Abstract: "{{abstract}}"

Based on the above observational proposal abstract, your task is to summarize the nature of the eventual observations. You will identify the astrophysical objects and phenomena, as well as the potential science use cases described in the abstract.

Follow these instructions exactly:
- Mention up to 5 items for both categories; do not mention more than 5 items in either category. 
- Choose the most relevant ones if there are more than 5 items in a category.
- Never mention the Hubble Space Telescope, HST, or the HST archive.
- Mention the class (e.g., barred spiral galaxy) and not just the specific instance (e.g., Andromeda).
- Name the objects in the science use cases, if appropriate.
- Write out full names of objects in addition to acronyms.
- Do not list irrelevant objects which do not describe the eventual observation, such as units or proposal Cycle numbers. List fewer but more relevant objects, if in doubt.
- Each science case listed must be self-contained but succinct.
- Only write in English.
- Do not list items that are too generic (e.g., galaxy, faint object, kinematics)
- The total length of text should not exceed 80 words.
- Present your lists in a comma-separated format; no dashed or numbered lists.

Example output: {'objects_and_phenomena':'spiral galaxies, galaxy clusters, supernova remnants', 'science_use_cases':'model galactic structure and evolution, characterize dark matter distribution in clusters, analyze expansion rates of supernova remnants'}

Answer in JSON format. The JSON should be a dictionary with keys "objects_and_phenomena" and "science_use_cases".

[/INST]
"""