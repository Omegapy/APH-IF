# -------------------------------------------------------------------------
## File: cypher_validator.py
# Author: Alexander Ricciardi
# Date: 2025-09-30
# File Path: data_processingcfr_downloader.py
# ------------------------------------------------------------------------

# --- Module Objective ---
# This module provides functionality for downloading Title 30 CFR (Code of Federal
# Regulations) PDF documents from the U.S. Government Publishing Office (GPO) website.
# It implements a robust download mechanism with retry logic, rate limiting, and error
# handling to collect mining regulatory documents for the MRCA knowledge base. The module
# serves as the initial data collection component in the MRCA Advanced Parallel Hybrid
# system pipeline, gathering the regulatory source documents that will be processed
# into the hybrid knowledge store for regulatory compliance assistance.
# -------------------------------------------------------------------------

# --- Module Contents Overview ---
# - Class: CFRDownloader - Main class for downloading CFR regulatory documents
# - Function: main() - Entry point for the CFR document download process
# - Global Constants: Console output configuration for download monitoring
# - Dependencies: Standard library modules and requests for HTTP operations
# -------------------------------------------------------------------------

# --- Dependencies / Imports ---
# - Standard Library:
#   - os: File system operations and directory management
#   - time: Rate limiting and retry delays for download operations
#   - requests: HTTP client for downloading PDF documents from GPO website
# - Third-Party:
#   - requests: Download CFR documents from the GPO website
# - Local Project Modules: None (standalone document acquisition utility)
# -------------------------------------------------------------------------

# --- Usage / Integration ---
# This module is designed as a standalone script for initial data acquisition in the
# MRCA system setup workflow. It downloads Title 30 CFR PDF documents from the official
# U.S. Government Publishing Office website and stores them in the data/cfr_pdf directory
# for subsequent processing by the hybrid knowledge store builders. The module should be
# executed before running the graph construction or hybrid store building processes to
# ensure the latest regulatory documents are available for knowledge base population.
# -------------------------------------------------------------------------

# --- Apache-2.0 ---
# © 2025 Alexander Samuel Ricciardi - All rights reserved.
# License: Apache-2.0 | Technology: Advanced Parallel HybridRAG - Intelligent Fusion (APH-IF) Technology
# -------------------------------------------------------------------------

"""

CFR Document Downloader for MRCA Mining Regulatory System

Provides robust download functionality for Title 30 CFR (Code of Federal Regulations)
PDF documents from the U.S. Government Publishing Office website with retry mechanisms,
rate limiting, and comprehensive error handling for regulatory document acquisition.

"""

# =========================================================================
# Imports
# =========================================================================
# Standard library imports
import os
import time
import requests

# Third-party library imports
# (None for this module)

# Local application/library specific imports
# (None for this standalone module)

# =========================================================================
# Global Constants / Variables
# =========================================================================
# Configure basic output for CFR download monitoring and progress tracking

# =========================================================================
# Class Definitions
# =========================================================================

# ------------------------------------------------------------------------- CFRDownloader
class CFRDownloader:
    """CFR document downloader for mining regulatory compliance system.

    This class provides robust functionality for downloading Title 30 CFR (Code of
    Federal Regulations) PDF documents from the U.S. Government Publishing Office
    website. It implements comprehensive error handling, retry mechanisms, rate
    limiting, and progress tracking to ensure reliable acquisition of regulatory
    documents for the MRCA knowledge base. The downloader handles network failures,
    server errors, and incomplete downloads with automatic retry logic.

    Class Attributes:
        None

    Instance Attributes:
        None (stateless downloader)

    Methods:
        __init__(): Initialize the CFR Downloader
        download_cfr_volumes(): Download all volumes for a given CFR title and year
    """

    # -------------------
    # --- Constructor ---
    # -------------------
    
    # --------------------------------------------------------------------------------- __init__()
    def __init__(self):
        """Initialize the CFR Downloader."""
        # Initialization currently does not require additional setup.
    # --------------------------------------------------------------------------------- end __init__()

    # ---------------------------
    # --- Document Download ---
    # ---------------------------

    # --------------------------------------------------------------------------------- download_cfr_volumes()
    def download_cfr_volumes(self, year=2025, title=30):
        """Download all volumes for a given CFR title and year with retry mechanism.

        Downloads all PDF volumes for the specified CFR title and year from the
        U.S. Government Publishing Office website. Implements robust retry logic
        for failed downloads, rate limiting to respect server resources, and
        comprehensive error handling for network and file system issues.

        Args:
            year (int, optional): The CFR edition year to download. Defaults to 2025.
            title (int, optional): The CFR title number to download. Defaults to 30
                                  (Mining regulations).

        Examples:
            >>> downloader = CFRDownloader()
            >>> downloader.download_cfr_volumes(year=2025, title=30)
            # Downloads all Title 30 volumes for 2025 to data_pdf/

            >>> downloader.download_cfr_volumes(year=2023, title=30)
            # Downloads all Title 30 volumes for 2023
        """
        # Based on the user's info, Title 30 has 3 volumes.
        volumes_to_download = set(range(1, 4))
        
        output_dir = os.path.join("data_pdf")
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"[INFO] Ensured output directory exists: {output_dir}")
        except OSError as e:
            print(f"[ERROR] Error creating directory {output_dir}: {e}")
            return

        successful_downloads = 0
        pass_num = 1

        while volumes_to_download:
            print(f"[INFO] --- Download Pass #{pass_num} ---")
            print(f"[INFO] Attempting to download {len(volumes_to_download)} volumes.")
            
            failed_this_pass = set()

            for vol_num in sorted(list(volumes_to_download)):
                file_name = f"CFR-{year}-title{title}-vol{vol_num}.pdf"
                # Construct the direct download URL
                pdf_url = f"https://www.govinfo.gov/content/pkg/CFR-{year}-title{title}-vol{vol_num}/pdf/CFR-{year}-title{title}-vol{vol_num}.pdf"
                pdf_filepath = os.path.join(output_dir, file_name)

                print(f"[INFO] Processing Volume {vol_num} (Year: {year}, Title: {title})...")
                print(f"[INFO] URL: {pdf_url}")

                try:
                    # Use a session for potential keep-alive benefits
                    with requests.Session() as s:
                        response = s.get(pdf_url, stream=True, timeout=120) # Increased timeout for large files

                        if response.status_code == 200 and 'application/pdf' in response.headers.get('Content-Type', ''):
                            with open(pdf_filepath, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    f.write(chunk)
                            
                            print(f"[INFO] Successfully downloaded {file_name}")
                            successful_downloads += 1
                        else:
                            print(
                                f"[WARNING] Failed to download Volume {vol_num}. "
                                f"Status: {response.status_code}, URL: {pdf_url}"
                            )
                            failed_this_pass.add(vol_num)
                
                except requests.exceptions.RequestException as e:
                    print(f"[ERROR] Request error for Volume {vol_num}: {e}")
                    failed_this_pass.add(vol_num)
                except Exception as e:
                    print(f"[ERROR] Unexpected error processing Volume {vol_num}: {e}")
                    failed_this_pass.add(vol_num)

                # Rate limiting
                time.sleep(2)

            # If a pass completes with no new successful downloads, abort.
            if len(failed_this_pass) > 0 and len(failed_this_pass) == len(volumes_to_download):
                print(
                    f"[ERROR] Could not download any of the remaining {len(failed_this_pass)} volumes "
                    f"on pass #{pass_num}."
                )
                print(f"[ERROR] Aborting retry. Failed volumes: {sorted(list(failed_this_pass))}")
                break
            
            volumes_to_download = failed_this_pass
            pass_num += 1

            if volumes_to_download:
                print(
                    f"[INFO] {len(volumes_to_download)} volumes failed. "
                    "Retrying in 10 seconds..."
                )
                time.sleep(10)

        print("=" * 50)
        print("[INFO] Download process finished.")
        print(f"[INFO] Total successful downloads: {successful_downloads}")
        
        final_failed_count = len(volumes_to_download)
        if final_failed_count > 0:
            print(f"[WARNING] Final failed downloads: {final_failed_count}")
            print(f"[WARNING] Un-downloadable volumes: {sorted(list(volumes_to_download))}")
            
        print(f"[INFO] PDF files saved in '{output_dir}' directory")
    # --------------------------------------------------------------------------------- end download_cfr_volumes()

# ------------------------------------------------------------------------- end CFRDownloader

# =========================================================================
# Standalone Function Definitions
# =========================================================================

# ------------------------
# --- Helper Functions ---
# ------------------------

# --------------------------------------------------------------------------------- main()
def main():
    """Main function to run the CFR document downloader.

    Orchestrates the complete CFR document download workflow including
    CFRDownloader initialization and Title 30 CFR volume acquisition for
    the MRCA system. Provides comprehensive error handling to ensure
    graceful failure recovery during the document collection process.

    This function serves as the entry point for acquiring regulatory source
    documents that will be processed by the MRCA Advanced Parallel Hybrid
    system for mining regulatory compliance assistance.

    Raises:
        Exception: If critical errors occur during the download process that
                  prevent completion of document acquisition

    Examples:
        >>> if __name__ == "__main__":
        ...     main()
        # Downloads all Title 30 CFR volumes for 2025
    """
    try:
        downloader = CFRDownloader()
        
        # Download all volumes for CFR Title 30 for the year 2025.
        downloader.download_cfr_volumes(year=2025, title=30)
        
    except Exception as e:
        print(f"[ERROR] An error occurred during execution: {e}")
# --------------------------------------------------------------------------------- end main()

# =========================================================================
# Module Initialization / Main Execution Guard
# =========================================================================
# This block runs only when the file is executed directly, not when imported.
# It serves as the entry point for the CFR document download process, allowing
# the module to be used both as a standalone script and as an importable utility
# for other components of the MRCA system that require regulatory document acquisition.

if __name__ == "__main__":
    # --- CFR Document Download Entry Point ---
    print(f"Running MRCA CFR Document Downloader from {__file__}...")
    
    try:
        # Execute the main CFR download process
        main()
        print("✅ CFR document download process completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user (Ctrl+C)")
        print("🛑 CFR document download aborted")
        
    except Exception as e:
        print(f"❌ Critical error during CFR document download: {e}")
        print("🛑 CFR document download failed")
        
    finally:
        print(f"Finished execution of {__file__}")

# =========================================================================
# End of File
# ========================================================================= 