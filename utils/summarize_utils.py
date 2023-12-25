import outlines
import outlines.models as models

from pydantic import BaseModel, Field, constr, conlist
from enum import Enum

class ConstrainedResponseHST(BaseModel):
    objects_and_phenomena: str
    science_use_cases: str

@outlines.prompt
def prompt_fn(abstract):
     """<|im_start|>system
You are an expert astrophysicist, with broad expertise across observational and theoretical astrophysics.
You are able to extract core information from astrophysical texts.

<|im_start|>user
Abstract: "{{abstract}}"

The above is an abstract for a proposed observation taken by the Hubble Space Telescope.

Your will summarize the nature of eventual observation taken by the Telescope, mentioning the astrophysical objects imaged and possible downstream science use cases described in the abstract.

Follow these instructions exactly:
- You will never mention the Hubble Space Telescope, HST, or the HST archive.
- You will only write text and numbers.
- You will not return empty values.
- You will only write in English.
- You will mention the class (e.g., barred spiral galaxy) and not just the specific instance (e.g., Andromeda).
- You will name every single astrophysical object, phenomenon, and possible science use case mentioned in the abstract.
- You can name the objects in the science use cases, if appropriate.
- You will be descriptive in your answers and list multiple items when appropriate. Do not used dashed or numbered lists, only comma-separated.
- The total length of text should not exceed 80 words.

Answer in JSON. The JSON should be a a dictionary with keys "objects_and_phenomena" and "science_use_cases".

<|im_start|>assistant
"""

