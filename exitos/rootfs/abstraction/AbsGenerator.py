# Class that is the parent for all different energy generators
from abstraction.AbsDevice import AbsDevice
from abc import abstractmethod


class AbsGenerator(AbsDevice):
    """
    Class that is the parent for all different energy generators (solar, wind, etc.)
    """

    def __init__(self, config):
        # Initialize the generator with the given configuration and name
        super().__init__(config)
        
        # Initialize min and max from restrictions if they exist
        if 'restrictions' in config and config['restrictions']:
            if 'min' in config['restrictions']:
                self.min = float(config['restrictions']['min']['value'])
            else:
                self.min = 0.0
                
            if 'max' in config['restrictions']:
                self.max = float(config['restrictions']['max']['value'])
            else:
                self.max = float('inf')
        else:
            # Default values if no restrictions are defined
            self.min = 0.0
            self.max = float('inf')
