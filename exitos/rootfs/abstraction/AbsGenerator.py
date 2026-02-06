# Class that is the parent for all different energy generators
from abstraction.AbsDevice import AbsDevice
from abc import abstractmethod
import logging

logger = logging.getLogger("exitOS")


class AbsGenerator(AbsDevice):
    """
    Class that is the parent for all different energy genators
    """

    def __init__(self, config):
        # Initialize the generator with the given configuration and name
        super().__init__(config)
        
        # Initialize min and max from restrictions if they exist
        if 'restrictions' in config and config['restrictions']:
            self.min = self._extract_numeric_value(config['restrictions'], 'min', default=0.0)
            self.max = self._extract_numeric_value(config['restrictions'], 'max', default=float('inf'))
        else:
            # Default values if no restrictions are defined
            self.min = 0.0
            self.max = float('inf')
        
        logger.debug(f"Initialized {self.name}: min={self.min} (type: {type(self.min)}), max={self.max} (type: {type(self.max)})")
    
    def _extract_numeric_value(self, restrictions, key, default=None):
        """
        Safely extracts a numeric value from restrictions.
        Handles both dict format {'value': X} and direct numeric values.
        """
        if key not in restrictions:
            if default is not None:
                return default
            raise ValueError(f"Missing '{key}' in restrictions for {self.name}")
        
        value = restrictions[key]
        
        # If it's a dict with 'value' key
        if isinstance(value, dict) and 'value' in value:
            return float(value['value'])
        
        # If it's a list, take the first element
        if isinstance(value, list):
            if len(value) > 0:
                logger.warning(f"⚠️ {key} is a list for {self.name}, taking first element: {value}")
                return float(value[0])
            elif default is not None:
                return default
            else:
                raise ValueError(f"{key} is an empty list for {self.name}")
        
        # If it's already a number
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            if default is not None:
                logger.warning(f"⚠️ Cannot convert {key}={value} to float for {self.name}, using default {default}: {e}")
                return default
            raise ValueError(f"Cannot convert {key}={value} (type: {type(value)}) to float for {self.name}: {e}")
