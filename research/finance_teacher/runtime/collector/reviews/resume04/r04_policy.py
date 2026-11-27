import time

MIN_REMAINING=190.

def budget_type(base):
    class FullWindowBudget(base):
        def admit(self,*args,**kwargs):
            if self.deadline-time.time()<MIN_REMAINING:
                raise ValueError('insufficient_full_180_second_request_window')
            return super().admit(*args,**kwargs)
    return FullWindowBudget

