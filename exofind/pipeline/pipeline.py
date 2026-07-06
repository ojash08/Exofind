"""
pipeline/pipeline.py — Main Orchestrator

Defines the ExoFind Pipeline, which orchestrates the
execution of various stages on a Target.
"""

from core.target import ExoFindTarget


class ExoFindPipeline:
    """
    Orchestrates the execution of multiple PipelineStages
    on a single ExoFindTarget.
    """

    def __init__(self, tpf_path):
        """
        Initialize the pipeline for a specific file.
        """
        self.target = ExoFindTarget(tpf_path)
        self.stages = []

    def add_stage(self, stage):
        """
        Add a PipelineStage to the execution list.
        """
        self.stages.append(stage)

    def run(self):
        """
        Run all configured stages on the target in sequence.
        """
        print("=" * 65)
        print(f"  ExoFind Pipeline — Target: {self.target.filename}")
        print("=" * 65)

        for stage in self.stages:
            print(f"\n--- Running {stage.name} ---")
            try:
                stage.execute(self.target)
            except Exception as e:
                print(f"  [ERROR] {stage.name} failed: {e}")
                import traceback
                traceback.print_exc()
                # Stop execution if a stage fails
                break

        print("\n" + "=" * 65)
        print("  Pipeline Execution Complete")
        print("=" * 65)

        return self.target
