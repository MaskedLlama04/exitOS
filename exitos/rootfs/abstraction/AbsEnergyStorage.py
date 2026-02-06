# Class that is the parent for all different energy storage devices
from abstraction.AbsDevice import AbsDevice
from abc import abstractmethod
import logging

logger = logging.getLogger("exitOS")


class AbsEnergyStorage(AbsDevice):
    """
    Class that is the parent for all different energy storage devices
    """

    def __init__(self, config):
        # Initialize the energy source with the given configuration and name
        super().__init__(config)
        
        # Safely extract min and max values
        self.min = self._extract_numeric_value(config['restrictions'], 'min')
        self.max = self._extract_numeric_value(config['restrictions'], 'max')
        
        logger.debug(f"Initialized {self.name}: min={self.min} (type: {type(self.min)}), max={self.max} (type: {type(self.max)})")
    
    def _extract_numeric_value(self, restrictions, key):
        """
        Safely extracts a numeric value from restrictions.
        Handles both dict format {'value': X} and direct numeric values.
        """
        if key not in restrictions:
            raise ValueError(f"Missing '{key}' in restrictions for {self.name}")
        
        value = restrictions[key]
        
        # If it's a dict with 'value' key
        if isinstance(value, dict) and 'value' in value:
            return float(value['value'])
        
        # If it's a list, take the first element (shouldn't happen, but handle it)
        if isinstance(value, list):
            if len(value) > 0:
                logger.warning(f"⚠️ {key} is a list for {self.name}, taking first element: {value}")
                return float(value[0])
            else:
                raise ValueError(f"{key} is an empty list for {self.name}")
        
        # If it's already a number
        try:
            return float(value)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert {key}={value} (type: {type(value)}) to float for {self.name}: {e}")


