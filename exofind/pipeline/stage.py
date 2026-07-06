"""
pipeline/stage.py — Base Pipeline Stage

Defines the abstract base class for all ExoFind pipeline stages.
"""

from core.target import ExoFindTarget

class PipelineStage:
    """
    Abstract base class for a stage in the ExoFind pipeline.
    """

    @property
    def name(self):
        """Returns the name of the stage."""
        return self.__class__.__name__

    def execute(self, target: ExoFindTarget):
        """
        Execute this stage on the given target.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement execute()")
