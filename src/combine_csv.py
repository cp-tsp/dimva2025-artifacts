import argparse
import pathlib
import subprocess
import sys
import pandas as pd


def get_csv_list(input_files, directory):
    if directory:
        return [directory / i for i in subprocess.check_output(["ls", f"{directory}"], text=True).split("\n")[:-1]]
    elif input_files:
        return input_files
    else:
        raise Exception("No input file given. See help.")


def combine_csv(input_files: [pathlib.Path], directory: pathlib.Path, output_path: pathlib.Path):
    """
    Combine all the input csv and add a label "Cobalt Strike" or "Benign" depending on the original filename
    :param input_files:
    :param directory:
    :param output_path:
    """
    csv_list = get_csv_list(input_files, directory)
    df = pd.DataFrame()
    for file in csv_list:
        d = pd.read_csv(file)
        if 'label' not in d:
            if "cs" in file.stem:
                d['label'] = "CobaltStrike"
                # d['label'] = file.stem.split("_")[1]
            else:
                d['label'] = "Benign"
        df = pd.concat([df, d], ignore_index=True)

    df.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_files", type=str, nargs="*")
    parser.add_argument("-d", "--directory", type=str, nargs="?")
    parser.add_argument("-o", "--output_path", type=str, default="", required=True)
    args = parser.parse_args()

    if args.input_files:
        input_files = [pathlib.Path(i) for i in args.input_files]
    else:
        input_files = []

    if args.directory:
        directory = pathlib.Path(args.directory)
    else:
        directory = None

    output_path = pathlib.Path(args.output_path)

    combine_csv(input_files, directory, output_path)


def run_main():
    sys.exit(main())


if __name__ == "__main__":
    run_main()
