"""
core/target.py — ExoFind Data Object

This module defines the ExoFindTarget class, which holds all state
for a single astronomical target as it moves through the pipeline.
"""

import os


class ExoFindTarget:
    """
    Holds all state and data for a single target through
    the ExoFind detection pipeline.
    """

    def __init__(self, tpf_path):
        """
        Initialize a target from a Target Pixel File path.
        """
        self.tpf_path = tpf_path
        
        # Extract filename for display purposes
        self.filename = os.path.basename(tpf_path)
        
        # We will parse the TIC ID from the FITS header later
        self.tic_id = None
        
        # --- Stage 1 & 2: Photometry ---
        self.tpf = None
        self.raw_time = None
        self.raw_flux = None
        self.raw_quality = None
        self.aperture_mask = None
        self.aperture_score = None
        
        # --- Stage 3: Cleaning ---
        self.clean_time = None
        self.clean_flux = None
        self.baseline = None
        
        # --- Stage 4: BLS Search ---
        self.bls_results = None
        
        # --- Stage 5: Vetting ---
        self.candidates = []
        self.best_candidate = None

    def __str__(self):
        name = f"TIC {self.tic_id}" if self.tic_id else self.filename
        return f"<ExoFindTarget: {name}>"
