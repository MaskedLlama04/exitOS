import logging
from datetime import datetime

from abstraction.DeviceRegistry import register_device
from abstraction.AbsEnergyStorage import AbsEnergyStorage

logger = logging.getLogger("exitOS")


@register_device("SonnenBattery")
class SonnenBattery(AbsEnergyStorage):
    # Inicialitzacio
    def __init__(self,config, database):
        super().__init__(config)

        # Obte la eficiencia i el percentatge actual
        efficiency_data = database.get_latest_data_from_sensor(config["extra_vars"]["eficiencia"]["sensor_id"])
        percentage_data = database.get_latest_data_from_sensor(config["extra_vars"]["percentatge_actual"]["sensor_id"])
        
        # Validate sensor data (check for None or zero/invalid values)
        if efficiency_data is None or efficiency_data[1] is None or float(efficiency_data[1]) <= 0:
            logger.warning(f"⚠️ Eficiència no disponible o invàlida per a {self.name} (valor: {efficiency_data[1] if efficiency_data else 'None'}), utilitzant valor per defecte 0.95")
            self.efficiency = 0.95
        else:
            self.efficiency = float(efficiency_data[1])
        
        if percentage_data is None or percentage_data[1] is None or float(percentage_data[1]) < 0:
            logger.warning(f"⚠️ Percentatge actual no disponible o invàlid per a {self.name} (valor: {percentage_data[1] if percentage_data else 'None'}), utilitzant valor per defecte 50%")
            self.actual_percentage = 0.5
        else:
            # Convert percentage to decimal (0-1 range)
            value = float(percentage_data[1])
            # If value is > 1, assume it's in percentage format (0-100)
            self.actual_percentage = value / 100.0 if value > 1 else value
        
        logger.info(f"   ▫️ {self.name}: efficiency={self.efficiency}, actual_percentage={self.actual_percentage}")

        # Obte els sensors de control
        self.control_charge_sensor = config['control_vars']['carregar']['sensor_id']
        self.control_discharge_sensor = config['control_vars']['descarregar']['sensor_id']
        self.control_mode_sensor = config['control_vars']['mode_operar']['sensor_id']

    def simula(self, config, horizon, horizon_min):
        kw_carrega = []  # Estat de càrrega (SoC) en cada moment
        consumption_profile = []  # El que realment consumeix/aporta la bateria
        total_cost = 0

        # Obte la capacitat actual
        actual_capacity_kwh = self.max * self.actual_percentage
        num_intervals = (horizon - 1) * horizon_min

        # Simula la bateria
        for i in range(num_intervals):
            accio_proposada = config[i]

            # Calculem el nou estat teòric
            if accio_proposada > 0:  # Carregant
                nou_estat = actual_capacity_kwh + (accio_proposada * self.efficiency)
            else:  # Descarregant
                nou_estat = actual_capacity_kwh + accio_proposada

            accio_real = accio_proposada
            cost_penalitzacio = 0

            # Control de límits (sense modificar el vector 'config' original)
            if nou_estat > self.max:
                cost_penalitzacio = (nou_estat - self.max) * 10  # Penalitzem l'excés
                accio_real = (self.max - actual_capacity_kwh) / self.efficiency if accio_proposada > 0 else 0
                actual_capacity_kwh = self.max
            elif nou_estat < self.min:
                cost_penalitzacio = (self.min - nou_estat) * 10  # Penalitzem descarregar massa
                accio_real = self.min - actual_capacity_kwh
                actual_capacity_kwh = self.min
            else:
                actual_capacity_kwh = nou_estat

            kw_carrega.append(actual_capacity_kwh)
            consumption_profile.append(accio_real)
            total_cost += cost_penalitzacio


        consumption_profile_24h = [0.0] * 24
        for i in range(min(len(consumption_profile), 23)):
            consumption_profile_24h[i + 1] = consumption_profile[i]

        # Retorna el perfil de consum, el cost total i la configuracio       
        return_dict = {
            "consumption_profile": consumption_profile_24h,
            "consumed_Kwh": kw_carrega,
            "total_cost": total_cost,
            "schedule": consumption_profile
        }

        return return_dict


    def controla(self, config,current_hour):

        # Obte el valor actual
        positive_value = abs(config[current_hour])
        value_to_HA = positive_value * 1000

        # Loggeja la configuracio actual
        if config[current_hour] >= 0: 
            # Carrega
            logger.info(f"     ▫️ Configurant {self.name} -> 🔋 Charge {value_to_HA}")
            return value_to_HA, self.control_charge_sensor, 'number'
        elif config[current_hour] < 0:
            # Descarrega
            logger.info(f"     ▫️ Configurant {self.name} -> 🪫 Discharge {value_to_HA}")
            return value_to_HA, self.control_discharge_sensor, 'number'

        return None
