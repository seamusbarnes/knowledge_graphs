#!/usr/bin/env python3
"""
remove_jupyter_markers.py

A CLI tool to remove Jupyter nbconvert cell markers and optionally all comments from a Python script.

Usage:
    python remove_jupyter_markers.py input.py [output.py] [--remove-comments]
"""

import argparse
import re
import sys
import os
import tempfile
import shutil

def is_jupyter_cell_marker(line):
    if re.match(r'^\s*#\s*In\[\s*\d*\s*\]\s*:\s*$', line):
        return True
    if re.match(r'^\s*#\s*%%\s*$', line):
        return True
    return False

def is_comment(line):
        # Remove lines starting with # (not inside string literals)
        s = line.strip()
        return s.startswith("# ")

def remove_jupyter_markers(input_path, output_path):
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not is_jupyter_cell_marker(line):
                fout.write(line)

def remove_comments(file_path):
    """Remove all comments (but not docstrings) from a Python file in place."""
    
    new_lines = []
    with open(file_path, "r", encoding="utf-8") as f:
        in_multiline_string = False
        for line in f:
            lstrip = line.lstrip()
            # Check for multiline string delimiters (""", ''')
            if lstrip.startswith('"""') or lstrip.startswith("'''"):
                if not in_multiline_string:
                    in_multiline_string = True
                elif in_multiline_string:
                    in_multiline_string = False
                new_lines.append(line)
                continue
            if in_multiline_string:
                new_lines.append(line)
                continue
            # Remove comments, but keep code before the comment
            if "#" in line and not is_comment(line):
                # Remove the comment part unless inside a string
                parts = re.split(r'(?<!["\'])#', line, maxsplit=1)
                new_lines.append(parts[0].rstrip() + "\n")
            elif not is_comment(line):
                new_lines.append(line)
            # else: skip pure comment lines
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def overwrite_file_safely(input_path, remove_comments_flag=False):
    dir_name = os.path.dirname(input_path)
    with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding="utf-8") as tmpfile:
        try:
            remove_jupyter_markers(input_path, tmpfile.name)
            tmpfile.flush()
            os.fsync(tmpfile.fileno())
            shutil.move(tmpfile.name, input_path)
            if remove_comments_flag:
                remove_comments(input_path)
            collapse_blank_lines(input_path)
            print(f"Cleaned file saved to '{input_path}'.")
        except Exception as e:
            print(f"Error during file overwrite: {e}", file=sys.stderr)
            os.unlink(tmpfile.name)
            sys.exit(1)

def collapse_blank_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                new_lines.append(line)
        else:
            blank_count = 0
            new_lines.append(line)

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

def main():
    parser = argparse.ArgumentParser(description="Remove Jupyter nbconvert cell markers and optionally all comments from a Python script.")
    parser.add_argument("input", help="Path to the input Python (.py) file generated by nbconvert.")
    parser.add_argument("output", nargs="?", help="Output cleaned Python file (or overwrite input if omitted).")
    parser.add_argument("--remove-comments", action="store_true", help="Remove all comments from the script (excluding docstrings).")
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.isfile(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        confirm = input(f"No output file specified. This will OVERWRITE '{input_path}'. Continue? [y/N]: ")
        if confirm.lower() != "y":
            print("Aborted.")
            sys.exit(0)
        overwrite_file_safely(input_path, remove_comments_flag=args.remove_comments)
    else:
        try:
            remove_jupyter_markers(input_path, output_path)
            if args.remove_comments:
                remove_comments(output_path)
            collapse_blank_lines(output_path)
            print(f"Cleaned file saved to '{output_path}'.")
        except Exception as e:
            print(f"Error writing output file: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
