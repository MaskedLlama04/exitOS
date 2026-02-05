import logging
from datetime import datetime

from abstraction.DeviceRegistry import register_device
from abstraction.AbsConsumer import AbsConsumer

logger = logging.getLogger("exitOS")


@register_device("CarCharger")
class CarCharger(AbsConsumer):
    """
    Carregador de vehicle elèctric (EV Charger).
    Simula la càrrega d'un cotxe elèctric amb control de potència variable.
    """

    def __init__(self, config, database):
        # Inicialitza la classe base
        super().__init__(config)
        
        # Inicialitza la base de dades
        self.database = database
        
        # Obté la capacitat de la bateria del cotxe (kWh)
        self.battery_capacity = float(config['restrictions']['capacitat_bateria']['value'])
        
        # Obté el percentatge actual de la bateria del cotxe
        self.current_battery_percentage = database.get_latest_data_from_sensor(
            config["extra_vars"]["percentatge_actual"]["sensor_id"]
        )[1]
        
        # Potència màxima i mínima de càrrega (kW)
        self.max_charging_power = float(config['restrictions']['potencia_maxima']['value'])
        self.min_charging_power = float(config['restrictions']['potencia_minima']['value'])
        
        # Eficiència de càrrega (típicament 85-95% per a carregadors AC)
        self.efficiency = 0.90
        
        # Límits de càrrega de la bateria
        self.min = self.min_charging_power
        self.max = self.max_charging_power
        
        # Sensor de control
        self.control_charging_power_sensor = config['control_vars']['potencia_carregar']['sensor_id']

    def simula(self, config, horizon, horizon_min):
        """
        Simula la càrrega del vehicle elèctric durant l'horitzó de planificació.
        
        Args:
            config: Vector de potències de càrrega planificades (kW)
            horizon: Horitzó de planificació (hores)
            horizon_min: Intervals per hora
            
        Returns:
            Diccionari amb perfil de consum, estat de càrrega i cost total
        """
        total_cost = 0
        self.horizon = horizon
        self.horizon_min = horizon_min
        consumption_profile = []
        battery_state_of_charge = []  # Percentatge de càrrega de la bateria
        
        # Estat inicial de la bateria
        current_battery_kwh = self.battery_capacity * (self.current_battery_percentage / 100.0)
        num_intervals = (horizon - 1) * horizon_min
        
        for i in range(num_intervals):
            charging_power = config[i]  # Potència de càrrega en kW
            
            # Calcular l'energia que s'afegeix a la bateria (considerant eficiència)
            # Energia = Potència * Temps (assumint intervals d'1 hora si horizon_min=1)
            time_interval = 1.0 / horizon_min  # Fracció d'hora
            energy_added = charging_power * time_interval * self.efficiency
            
            # Actualitzar l'estat de càrrega
            new_battery_kwh = current_battery_kwh + energy_added
            
            # Aplicar límits de la bateria
            cost_penalty = 0
            actual_charging_power = charging_power
            
            if new_battery_kwh > self.battery_capacity:
                # Bateria plena - penalitzar sobrecàrrega
                cost_penalty = (new_battery_kwh - self.battery_capacity) * 5
                # Ajustar la potència real per no sobrepassar
                max_energy_needed = self.battery_capacity - current_battery_kwh
                actual_charging_power = max_energy_needed / (time_interval * self.efficiency)
                new_battery_kwh = self.battery_capacity
            elif new_battery_kwh < 0:
                # No es pot descarregar (aquest dispositiu només carrega)
                cost_penalty = abs(new_battery_kwh) * 5
                actual_charging_power = 0
                new_battery_kwh = current_battery_kwh
            
            # Actualitzar estat
            current_battery_kwh = new_battery_kwh
            current_percentage = (current_battery_kwh / self.battery_capacity) * 100.0
            
            # Guardar resultats
            battery_state_of_charge.append(current_percentage)
            consumption_profile.append(actual_charging_power)
            total_cost += cost_penalty

        # Convertir a perfil de 24 hores (si cal)
        consumption_profile_24h = [0.0] * 24
        for i in range(min(len(consumption_profile), 23)):
            consumption_profile_24h[i + 1] = consumption_profile[i]

        return_dict = {
            "consumption_profile": consumption_profile_24h,
            "battery_soc": battery_state_of_charge,
            "total_cost": total_cost,
            "schedule": consumption_profile
        }

        return return_dict

    def controla(self, config, current_hour):
        """
        Tradueix la planificació en comandes per a Home Assistant.
        
        Args:
            config: Vector de configuració completa
            current_hour: Hora actual
            
        Returns:
            Tupla (valor, sensor_id, tipus) per enviar a Home Assistant
        """
        # Obtenir la potència de càrrega planificada per a l'hora actual
        charging_power = config[current_hour]
        
        # Convertir a Watts (Home Assistant normalment treballa en W)
        value_to_HA = charging_power * 1000
        
        # Loggeja la configuració actual
        if charging_power > 0:
            logger.info(f"     ▫️ Configurant {self.name} -> 🔌 Charging {value_to_HA}W ({charging_power}kW)")
        else:
            logger.info(f"     ▫️ Configurant {self.name} -> ⏸️ Not charging")
        
        return value_to_HA, self.control_charging_power_sensor, 'number'
