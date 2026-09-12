
from typing import Any


class StateRecorder:
    
    record_points: list[int]|None
    
    def __init__(self, record_points: list[int]|None=None):
        
        self.record_points = record_points.copy() if record_points is not None else None
        self.state_list = {}
        
        
    def should_record(self, index: int) -> bool:
        
        return self.record_points is None or index in self.record_points
    
    def record(self, index: int, state: Any) -> None:
        
        self.state_list[index] = state
    
