import logging

from abstraction.DeviceRegistry import register_device
from abstraction.AbsGenerator import AbsGenerator

logger = logging.getLogger("exitOS")


@register_device("PV")
class PV(AbsGenerator):
    """
    Panells fotovoltaics (Photovoltaic panels).
    Genera energia basant-se en la radiació solar prevista.
    """
    
    def __init__(self, config, database):
        super().__init__(config)
        
        self.database = database
        self.max_output = 2500  # W
        self.min_output = 0
        self.eficiencia = 0.97

        self.numero_plaques = 10
        self.superficie_placa = 1.7
        self.superficie_total = self.numero_plaques * self.superficie_placa
        
        # Set min/max for optimization
        self.min = 0
        self.max = self.max_output / 1000  # Convert to kW

    def simula(self, config, horizon, horizon_min):
        """
        Simula la generació fotovoltaica.
        Nota: La generació solar és passiva i depèn de la radiació prevista,
        no de la configuració de l'optimitzador.
        
        Args:
            config: Vector de configuració (no utilitzat per a PV)
            horizon: Horitzó de planificació (hores)
            horizon_min: Intervals per hora
            
        Returns:
            Diccionari amb perfil de generació
        """
        # Per a PV, la generació és passiva i ve determinada per la radiació solar
        # La configuració no afecta la generació (no es pot "controlar" el sol)
        
        num_intervals = (horizon - 1) * horizon_min
        generation_profile = [0.0] * num_intervals
        
        # En un sistema real, aquí s'utilitzaria la previsió de radiació solar
        # Per ara retornem zeros ja que la generació real es gestiona externament
        
        return_dict = {
            "consumption_profile": generation_profile,  # Negatiu indicaria generació
            "total_cost": 0,  # La generació solar no té cost
            "schedule": config
        }
        
        return return_dict

    def controla(self, config, current_hour):
        """
        Control per a panells fotovoltaics.
        Nota: Els panells solars no es poden controlar activament,
        la seva generació depèn de la radiació solar.
        
        Args:
            config: Vector de configuració
            current_hour: Hora actual
            
        Returns:
            None (no hi ha control actiu per a PV)
        """
        # Els panells solars no necessiten control actiu
        logger.debug(f"     ▫️ {self.name} -> ☀️ Generació passiva (sense control)")
        return None

    def get_generacio_horaria(self, hourly_radiation):
        """
        Calcula la generació horària basant-se en la radiació solar.
        
        Args:
            hourly_radiation: Llista de valors de radiació solar per hora
            
        Returns:
            Llista amb la generació horària en kW
        """
        generacio_horaria_total = []
        for hour in hourly_radiation:
            calcul_aux = (hour * self.superficie_total * self.eficiencia) / 1000  # kW
            generacio_hora = min(calcul_aux, self.max_output / 1000)
            generacio_horaria_total.append(round(generacio_hora, 2))

        return generacio_horaria_total
