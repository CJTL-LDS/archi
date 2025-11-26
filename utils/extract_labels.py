import os
import argparse

def extract_and_write_labels(png_folder, output_file):
    """
    Extracts filenames from a folder and writes them to a text file.

    Args:
        png_folder (str): The path to the folder containing PNG files.
        output_file (str): The path to the output text file.
    """
    try:
        # Get all file names from the specified folder
        filenames = os.listdir(png_folder)
    except FileNotFoundError:
        print(f"Error: The directory '{png_folder}' was not found.")
        return

    try:
        # Write each filename to the output file
        with open(output_file, 'w', encoding='utf-8') as f:
            for filename in filenames:
                name_without_extension, _ = os.path.splitext(filename)
                f.write(f"{name_without_extension}\n")
        print(f"Successfully wrote {len(filenames)} labels to '{output_file}'.")
    except IOError as e:
        print(f"Error writing to file '{output_file}': {e}")

def main():
    parser = argparse.ArgumentParser(description="Extract filenames from a folder and save them to a .txt file.")
    parser.add_argument("--png_folder", type=str, default="png", help="Folder containing the source PNG files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .txt file.")
    
    args = parser.parse_args()

    extract_and_write_labels(args.png_folder, args.output_file)

if __name__ == "__main__":
    main()
