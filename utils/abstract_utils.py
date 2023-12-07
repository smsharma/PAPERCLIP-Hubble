import pandas as pd


def read_abstracts_file(filename):
    """ Read and parse the abstracts.cat file.
    """
    abstracts = []
    abstract = {}
    last_prop = None

    with open(filename, "r") as file:
        for line in file:
            stripped_line = line.strip()

            if stripped_line.startswith("-----"):
                if abstract:  # If abstract has content, append to list
                    abstracts.append(abstract)
                    abstract = {}
                    last_prop = None  # Reset the last property
            else:
                # Check for known properties
                property_starts = {"Prop. Type:": "Prop. Type", "Category:": "Category", "ID:": "ID", "Cycle:": "Cycle", "Title:": "Title", "PI:": "PI"}

                found_prop = None
                for prop_start, prop_name in property_starts.items():
                    if stripped_line.startswith(prop_start) and last_prop != "Abstract":
                        found_prop = prop_name
                        break

                if found_prop:
                    value = stripped_line.split(":", 1)[1].strip()
                    if found_prop in ["ID", "Cycle"]:
                        try:
                            value = int(value)
                        except ValueError:
                            pass

                    abstract[found_prop] = value
                    last_prop = found_prop  # Update the last property
                else:
                    # If none of the known properties are found,
                    # we treat the line as part of the last property or abstract.
                    if last_prop == "PI":
                        # After the PI line, the abstract starts
                        last_prop = "Abstract"
                        abstract[last_prop] = stripped_line
                    elif last_prop:
                        abstract[last_prop] += " " + stripped_line

    # After loop ends, check if there's any remaining content in the abstract dictionary
    if abstract:
        abstracts.append(abstract)

    df = pd.DataFrame(abstracts)
    return df
