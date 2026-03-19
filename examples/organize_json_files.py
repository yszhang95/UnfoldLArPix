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
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Set


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
    - 0-.*_sp(?:\d+)_spp(?:\d+)_t\d+p\d+\.json$
    - 0-.*_sp(?:\d+)_spp(?:\d+)_t\d+p\d+_smeared\.json$
    
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
            if identifier not in groups:
                groups[identifier] = []
            groups[identifier].append(json_file)
            continue
        
        # Try matching smeared pattern
        match = pattern_smeared.match(filename)
        if match:
            identifier = match.group(1)
            if identifier not in groups:
                groups[identifier] = []
            groups[identifier].append(json_file)
    
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
    - Copies both JSON files to the subdirectory
    
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
        
        # Copy files
        for json_file in files:
            dest_file = pattern_dir / json_file.name
            
            if dry_run:
                print(f"[DRY RUN] Would copy: {json_file} -> {dest_file}")
            else:
                shutil.copy2(json_file, dest_file)
                print(f"Copied: {json_file.name} -> {dest_file}")


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
    
    print(f"\nProcessed {len(groups)} pattern groups successfully")


if __name__ == "__main__":
    main()
