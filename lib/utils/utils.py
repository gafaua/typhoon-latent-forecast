from typing import List


class SimpleScheduler:
    def __init__(self,
                 value_range: List,
                 warmup: int,
                 last: int,
                 verbose=False):
        self.value_range = value_range
        self.warmup = warmup
        self.last  = last

        self.current_step = 0
        self.current_value = value_range[0]
        self.range = value_range[1] - value_range[0]
        self.step_range = self.last - self.warmup
        self.verbose = verbose

    def step(self):
        self.current_step += 1
        if self.current_step < self.warmup or self.current_step > self.last:
            return

        prop = (self.current_step - self.warmup) / self.step_range
        old_value = self.current_value
        self.current_value = prop * self.range + self.value_range[0]

        if isinstance(self.value_range[0], int):
            self.current_value = round(self.current_value)

        if self.verbose and self.current_value != old_value:
            print(f"New Value: {self.current_value}")

    def get_value(self):
        return self.current_value
