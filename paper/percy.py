import re
import argparse


def is_environment_start(line):
    return "\\begin{" in line


def is_environment_end(line):
    return "\\end{" in line


def modify_tex(tex_content):
    """
    Modifies the given TeX content by inserting a '%' symbol followed by a newline
    after each sentence within text sections, avoiding modification within environments.
    """
    modified_lines = []
    skip_environment = False

    for line in tex_content.split("\n"):
        if is_environment_start(line):
            skip_environment = True
        elif is_environment_end(line):
            skip_environment = False
            modified_lines.append(line)
            continue

        if not skip_environment:
            # Simple regex to match period followed by a space to replace with period, newline, and '%\n'
            line = re.sub(r"(\.)(\s)", r"\1\n%\n", line)
        modified_lines.append(line)

    return "\n".join(modified_lines)


def process_tex_file(input_file_path, output_file_path):
    """
    Reads, modifies, and saves the TeX file. The modified content is saved to output_file_path.
    """
    with open(input_file_path, "r", encoding="utf-8") as file:
        content = file.read()

    modified_content = modify_tex(content)

    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(modified_content)


def main():
    parser = argparse.ArgumentParser(
        description='Modify a TeX document by inserting a "%" symbol after each sentence within text sections and save the modified document optionally to a different file.'
    )
    parser.add_argument(
        "input_file_path", type=str, help="The path to the TeX document to be modified."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="The path where the modified TeX document will be saved. If not provided, the input file will be overwritten.",
        default=None,
    )

    args = parser.parse_args()

    output_file_path = args.output if args.output else args.input_file_path

    process_tex_file(args.input_file_path, output_file_path)
    print(f"Processed {args.input_file_path} and saved to {output_file_path}")


if __name__ == "__main__":
    main()
