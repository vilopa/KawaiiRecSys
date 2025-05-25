#!/usr/bin/env python
import os
import sys
import pstats
import glob
import argparse
from datetime import datetime

def list_profile_files():
    """List all profile files in the profiles directory."""
    if not os.path.exists("profiles"):
        print("Profiles directory not found. No profiling data available.")
        return []
    
    files = glob.glob("profiles/*.prof")
    if not files:
        print("No profile files found in the profiles directory.")
        return []
    
    files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"Found {len(files)} profile files:")
    for i, file in enumerate(files, 1):
        timestamp = os.path.getmtime(file)
        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{i}. {os.path.basename(file)} - {date_str}")
    
    return files

def analyze_profile(profile_path, sort_by='cumtime', limit=20):
    """Analyze a profile file and print the results."""
    if not os.path.exists(profile_path):
        print(f"Profile file not found: {profile_path}")
        return
    
    print(f"\nAnalyzing profile: {profile_path}")
    print(f"Sorting by: {sort_by}")
    print("-" * 80)
    
    # Create Stats object
    stats = pstats.Stats(profile_path)
    
    # Print summary
    stats.strip_dirs().sort_stats(sort_by).print_stats(limit)
    
    # Print callers of the most time-consuming functions
    print("\nTop 5 time-consuming function callers:")
    stats.sort_stats(sort_by)
    for func in [f[0] for f in stats.stats.items()][:5]:
        print(f"\nCallers of {func}:")
        stats.print_callees(func)

def main():
    """Main function to parse arguments and analyze profile files."""
    parser = argparse.ArgumentParser(description="Analyze cProfile results for KawaiiRecSys")
    parser.add_argument("-f", "--file", help="Specific profile file to analyze")
    parser.add_argument("-s", "--sort", default="cumtime", 
                        choices=["cumtime", "time", "calls", "pcalls", "tottime"],
                        help="Sort results by this metric")
    parser.add_argument("-l", "--limit", type=int, default=20,
                        help="Limit number of functions displayed")
    args = parser.parse_args()
    
    if args.file:
        # Analyze specific file
        if os.path.exists(args.file):
            analyze_profile(args.file, args.sort, args.limit)
        else:
            print(f"Profile file not found: {args.file}")
            files = list_profile_files()
            if files:
                choice = input("\nEnter the number of the file to analyze (or press Enter to exit): ")
                if choice and choice.isdigit() and 1 <= int(choice) <= len(files):
                    analyze_profile(files[int(choice) - 1], args.sort, args.limit)
    else:
        # List files and prompt for selection
        files = list_profile_files()
        if files:
            choice = input("\nEnter the number of the file to analyze (or press Enter to exit): ")
            if choice and choice.isdigit() and 1 <= int(choice) <= len(files):
                analyze_profile(files[int(choice) - 1], args.sort, args.limit)
    
if __name__ == "__main__":
    main() 