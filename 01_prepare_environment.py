# 01_prepare_environment.py

from src.config import (
    ROOT_DATA_FOLDER, MERGED_TILES_FOLDER, FINAL_PATCHES_FOLDER
)
from src.data_preprocessing import run_merge_step, run_patch_step
from src.data_splitter import create_and_save_splits

def run_data_preparation_and_config_generation():
    """
    Orchestrates data patching and dynamic config file generation.
    """
    print("--- STEP 1A: PREPARING DATA ---")
    
    # --- MERGE STEP ---
    # To skip merging, simply comment out the line below.
    run_merge_step(root_input_dir=ROOT_DATA_FOLDER, merged_output_dir=MERGED_TILES_FOLDER)
    
    # --- PATCHING STEP ---
    # This will now run independently, using the files from the merge step.
    run_patch_step(merged_input_dir=MERGED_TILES_FOLDER, patches_output_dir=FINAL_PATCHES_FOLDER, force_rerun=True)

def main():
    """Main execution function."""
    run_data_preparation_and_config_generation()
    
    # After preparing data, create the consistent splits
    create_and_save_splits()

if __name__ == "__main__":
    main()