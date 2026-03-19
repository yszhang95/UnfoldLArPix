#!/usr/bin/env python3
"""
Organize JSON files by extracting pattern-based identifiers and copying them
to structured destination directories.

This script:
1. Scans a source directory for JSON files
2. Filters files matching specific patterns (regular and _smeared versions)
3. Extracts pattern identifiers from filenames
4. Creates organized subdirectories in the destination
5. Copies matched JSON pairs to their respective subdirectories
6. Writes a "create_eventsets.py" script into the destination directory that is
   intended to be piped into an interactive Python shell to create EventSet
   objects for each matched subdirectory (alias/desc set to the subdir name).

The generated create_eventsets.py is designed to be executed by piping it into
an interactive Python interpreter, e.g.:

    cat create_eventsets.py | python -i

This allows the created Django objects to be inspected interactively after
the script runs.
"""

import argparse
import re
import shutil
import stat
from pathlib import Path
from typing import Dict, List


def cache_json_files(source_dir: Path) -> List[Path]:
    """
    Cache all JSON file paths under the source directory.

    Parameters
    ----------
    source_dir : Path
        Source directory to scan for JSON files

    Returns
    -------
    List[Path]
        List of all JSON file paths found
    """
    json_files = list(source_dir.glob("**/*.json"))
    print(f"Found {len(json_files)} JSON files in {source_dir}")
    return json_files


def extract_pattern_groups(json_files: List[Path]) -> Dict[str, List[Path]]:
    """
    Filter JSON files by pattern and group them by extracted identifier.

    Matches files with patterns:
    - 0-.*_sp(?:\\d+)_spp(?:\\d+)_t\\d+p\\d+\\.json$
    - 0-.*_sp(?:\\d+)_spp(?:\\d+)_t\\d+p\\d+_smeared\\.json$

    Parameters
    ----------
    json_files : List[Path]
        List of JSON file paths to filter

    Returns
    -------
    Dict[str, List[Path]]
        Dictionary mapping pattern identifiers to lists of matched files
    """
    # Patterns to match
    pattern_regular = re.compile(r'^0-(.*_sp\d+_spp\d+_t\d+p\d+)\.json$')
    pattern_smeared = re.compile(r'^0-(.*_sp\d+_spp\d+_t\d+p\d+)_smeared\.json$')

    groups: Dict[str, List[Path]] = {}

    for json_file in json_files:
        filename = json_file.name

        # Try matching regular pattern
        match = pattern_regular.match(filename)
        if match:
            identifier = match.group(1)
            groups.setdefault(identifier, []).append(json_file)
            continue

        # Try matching smeared pattern
        match = pattern_smeared.match(filename)
        if match:
            identifier = match.group(1)
            groups.setdefault(identifier, []).append(json_file)

    print(f"Found {len(groups)} unique pattern groups")
    return groups


def validate_groups(groups: Dict[str, List[Path]]) -> None:
    """
    Validate that each group has exactly two files (regular and smeared).

    Parameters
    ----------
    groups : Dict[str, List[Path]]
        Dictionary mapping pattern identifiers to lists of matched files

    Raises
    ------
    ValueError
        If any group doesn't have exactly 2 files
    """
    for identifier, files in groups.items():
        if len(files) != 2:
            file_list = ", ".join(f.name for f in files)
            raise ValueError(
                f"Expected 2 files for pattern '{identifier}', "
                f"found {len(files)}: {file_list}"
            )

        # Check that we have one regular and one smeared
        filenames = {f.name for f in files}
        expected_regular = f"0-{identifier}.json"
        expected_smeared = f"0-{identifier}_smeared.json"

        if expected_regular not in filenames:
            raise ValueError(
                f"Missing regular file '{expected_regular}' for pattern '{identifier}'"
            )
        if expected_smeared not in filenames:
            raise ValueError(
                f"Missing smeared file '{expected_smeared}' for pattern '{identifier}'"
            )


def organize_files(
    groups: Dict[str, List[Path]],
    dest_dir: Path,
    dry_run: bool = False
) -> None:
    """
    Create organized directory structure and copy files.

    For each pattern identifier:
    - Creates subdirectory: dest_dir/{identifier}/
    - Creates nested directories: data/0/
    - Copies both JSON files to data/0/ inside the subdirectory

    Parameters
    ----------
    groups : Dict[str, List[Path]]
        Dictionary mapping pattern identifiers to lists of matched files
    dest_dir : Path
        Destination directory for organized files
    dry_run : bool, optional
        If True, only print actions without executing them
    """
    for identifier, files in groups.items():
        # Create subdirectory structure
        pattern_dir = dest_dir / identifier
        data_dir = pattern_dir / "data" / "0"

        if dry_run:
            print(f"[DRY RUN] Would create directory: {pattern_dir}")
            print(f"[DRY RUN] Would create directory: {data_dir}")
        else:
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Created directory structure: {data_dir}")

        # Copy files into data/0
        for json_file in files:
            dest_file = data_dir / json_file.name

            if dry_run:
                print(f"[DRY RUN] Would copy: {json_file} -> {dest_file}")
            else:
                shutil.copy2(json_file, dest_file)
                print(f"Copied: {json_file.name} -> {dest_file}")


def write_create_eventsets_script(
    dest_dir: Path,
    identifiers: List[str],
    dry_run: bool = False
) -> None:
    """
    Write a script into dest_dir/create_eventsets.py that, when piped into an
    interactive Python shell, will create an EventSet for each identifier.

    The generated script contains top-level code (no main guard) so that it
    executes immediately when piped into the interpreter. Example usage:

        cat create_eventsets.py | python -i

    The script will attempt to create an EventSet for each identifier and will
    print results or errors for each attempt.
    """
    script_path = dest_dir / "create_eventsets.py"

    # Build identifiers list literal safely
    safe_idents = []
    for identifier in sorted(identifiers):
        # escape backslashes and quotes for safe literal embedding
        safe = identifier.replace("\\", "\\\\").replace('"', '\\"')
        safe_idents.append(f'"{safe}"')
    idents_literal = "[" + ", ".join(safe_idents) + "]"

    script_lines = [
        "# Generated script intended to be piped into an interactive Python shell.",
        "# Usage example: cat create_eventsets.py | python -i",
        "from events.models import EventSet",
        "from django.utils import timezone",
        "",
        f"IDENTIFIERS = {idents_literal}",
        "",
        "for subdir in IDENTIFIERS:",
        "    try:",
        "        eventset = EventSet.objects.create(",
        "            event_type='positron',",
        "            num_events=1,",
        "            energy='3GeV',",
        "            geometry='2x2',",
        "            desc=subdir,",
        "            alias=subdir,",
        "            created_at=timezone.now(),",
        "        )",
        "        print(f'Created EventSet id={eventset.id}, alias={eventset.alias}')",
        "    except Exception as e:",
        "        import traceback",
        "        traceback.print_exc()",
        "        print(f'Failed to create EventSet for {subdir}: {e}')",
        "",
        "# End of generated script",
        ""
    ]

    script_content = "\n".join(script_lines)

    if dry_run:
        print(f"[DRY RUN] Would write create_eventsets script to: {script_path}")
        print("--- Script content start ---")
        print(script_content)
        print("--- Script content end ---")
    else:
        dest_dir.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script_content)
        # Make the script readable/executable for convenience (execution via piping
        # doesn't require the executable bit, but set it anyway).
        mode = script_path.stat().st_mode
        script_path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        print(f"Wrote create_eventsets script to: {script_path}")
        print("To execute (piped into an interactive Python shell):")
        print(f"  cat {script_path} | python -i")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Organize JSON files by pattern into structured directories"
    )
    parser.add_argument(
        "source_dir",
        type=Path,
        help="Source directory containing JSON files"
    )
    parser.add_argument(
        "dest_dir",
        type=Path,
        help="Destination directory for organized files"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation that each pattern has exactly 2 files"
    )

    args = parser.parse_args()

    # Validate source directory exists
    if not args.source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {args.source_dir}")

    if not args.source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {args.source_dir}")

    # Cache all JSON files
    json_files = cache_json_files(args.source_dir)

    if not json_files:
        print("No JSON files found. Exiting.")
        return

    # Extract and group by pattern
    groups = extract_pattern_groups(json_files)

    if not groups:
        print("No files matching the specified patterns. Exiting.")
        return

    # Validate groups
    if not args.skip_validation:
        try:
            validate_groups(groups)
            print("Validation passed: all groups have exactly 2 files")
        except ValueError as e:
            print(f"Validation error: {e}")
            return

    # Organize files
    if not args.dry_run:
        args.dest_dir.mkdir(parents=True, exist_ok=True)

    organize_files(groups, args.dest_dir, dry_run=args.dry_run)

    # After organizing files, write the create_eventsets.py script into dest_dir.
    # The generated script is top-level so it can be piped into an interactive
    # Python shell (see write_create_eventsets_script docstring).
    identifiers = list(groups.keys())
    write_create_eventsets_script(args.dest_dir, identifiers, dry_run=args.dry_run)

    print(f"\nProcessed {len(groups)} pattern groups successfully")
    if not args.dry_run:
        print(f"A script to create EventSet entries was written to: {args.dest_dir / 'create_eventsets.py'}")


if __name__ == "__main__":
    main()
