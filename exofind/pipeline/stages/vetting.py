"""
pipeline/stages/vetting.py — Stage 5

Handles Candidate Analysis and False Positive rejection.
Checks for odd/even transit depth mismatch and secondary eclipses.
"""

from pipeline.stage import PipelineStage
from vetting.odd_even import check_odd_even_mismatch
from vetting.secondary_eclipse import check_secondary_eclipse

class VettingStage(PipelineStage):

    def execute(self, target):
        
        if not target.candidates:
            print("  No candidates to vet.")
            return
            
        print(f"  Vetting {len(target.candidates)} candidates...")
        
        for cand in target.candidates:
            
            # 1. Harmonic check (already flagged in Stage 4)
            if cand.get("is_harmonic"):
                cand["status"] = "REJECTED (Harmonic Alias)"
                continue
                
            # 2. Odd/Even Mismatch
            oe_res = check_odd_even_mismatch(
                target.clean_time,
                target.clean_flux,
                cand["period"],
                cand["phase_start"],
                cand["phase_width"]
            )
            
            cand["odd_even_diff_sigma"] = oe_res.get("diff_sigma", 0)
            
            if oe_res.get("is_false_positive"):
                cand["status"] = "REJECTED (Odd/Even Mismatch)"
                continue
                
            # 3. Secondary Eclipse
            se_res = check_secondary_eclipse(
                target.clean_time,
                target.clean_flux,
                cand["period"],
                cand["phase_start"],
                cand["phase_width"],
                abs(cand["depth"])
            )
            
            cand["secondary_ratio"] = se_res.get("ratio_to_primary", 0)
            
            if se_res.get("is_false_positive"):
                cand["status"] = "REJECTED (Secondary Eclipse)"
                continue
                
            # If it passes all tests
            cand["status"] = "PASSED"
            
        # Select best passing candidate
        passed = [c for c in target.candidates if c.get("status") == "PASSED"]
        if passed:
            # They are already sorted by SDE descending from search stage
            target.best_candidate = passed[0]
            print(f"  Vetting complete. Best candidate period: {target.best_candidate['period']:.4f} days")
        else:
            print("  Vetting complete. ALL candidates rejected as false positives.")
            target.best_candidate = None
