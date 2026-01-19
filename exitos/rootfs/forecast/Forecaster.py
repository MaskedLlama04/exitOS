from datetime import timedelta

import joblib
import numpy as np
import pandas as pd
import holidays
import logging
import os
import glob
import requests

logger = logging.getLogger("exitOS")


class Forecaster:
    def __init__(self, debug=False):
        """
        Constructor per defecte del Forecaster
        """

        self.debug = debug
        self.search_space_config_file = 'resources/search_space.conf'
        self.db = dict()

        if "HASSIO_TOKEN" in os.environ:
            self.models_filepath = "share/exitos/"
        else:
            self.models_filepath = "./share/exitos/"

    @staticmethod
    def windowing_group(dataset, look_back_start=24, look_back_end=48):
        """
        Funció per crear les variables del windowing. \n
        Treballa sobre un dataset i inclou la variable objectiu. \n
        Les variables creades es diràn com les originals (legacy) i s'afagirà '_' amb el número de desplaçament. \n
        Es tindràn en compte les hores en el rang [ini,fi)

        :param dataset: Dataframe amb datetime com a índex
        :type dataset: pd.DataFrame
        :param look_back_start: On comença la finestra ( 24 -> el dia anterior si és horari)
        :param look_back_end: On acaba el número d'observacions (48 -> el dia anterior si és horari)
        :return: Dataset amb les cariables desplaçades en columnes noves
        :rtype: pd.DataFrame
        """

        ds = dataset.copy()
        shifted_columns = {}

        for col in ds.columns:
            if col != 'timestamp':
                for j in range(look_back_start, look_back_end):
                    shifted_columns[f"{col}_{j}"] = ds[col].shift(j)

        ds = pd.concat([ds, pd.DataFrame(shifted_columns, index=ds.index)], axis=1)

        return ds

    @staticmethod
    def windowing_univariant(dataset, look_back_start=24, look_back_end=48, variable=''):
        """
        Funció per crear les variables del windowing. \n
        Treballa sobre un dataset i inclou la variable objectiu. \n
        Les variables creades es diràn com les originals (legacy) i s'afagirà '_' amb el número de desplaçament. \n
        Es tindràn en compte les hores en el rang [ini,fi)

        :param dataset: Dataframe amb datetime com a índex
        :type dataset: pd.DataFrame
        :param look_back_start: On comença la finestra ( 24 -> el dia anterior si és horari)
        :param look_back_end: On acaba el número d'observacions (48 -> el dia anterior si és horari)
        :param variable: Variable a transformar en variables del windowing.
        :return: Dataset amb les cariables desplaçades en columnes noves
        :rtype: pd.DataFrame
        """

        ds = dataset.copy()
        for i in range(0, len(ds.columns)):
            if ds.columns[i] == variable:
                for j in range(look_back_start, look_back_end - 1):
                    ds[ds.columns[i] + '_' + str(j)] = ds[ds.columns[i]].shift(j)

        return ds

    def do_windowing(self, data, look_back={-1: [25, 48]}):
        """
        Aplica el Windowing en consequencia al look_back indicat.\n
        - None -> no aplica el windowing \n
        - Diccionari on la clau és la variable a fer windowing i el valor la finestra que s'ha d'aplicar \n
        - Les claus son Strings, indicant el nom de la columna a aplicar windowing
        - Si com a clau es dona -1, la finestra aplicara a totes les variables NO especificades individualment.
        - Els valors són els que defineixen la finestra a aplicar, i poden ser:
            - [ini, fi]
            - [ini, fi, salt]
        :param data: dataframe de dades
        :param look_back: Windowing a aplicar
        :return: dataframe de dades preparades per el model de forecasting.
        """
        if look_back is not None:
            # windowing  de tots els grups (NO individuals)
            if -1 in look_back.keys():  # si l'indicador és -1 volem un grup
                aux = look_back[-1]  # recuperem els valors de la finestra pel grup

                # recuperem les que es faran soles
                keys = list()
                for i in look_back.keys():
                    if i != -1:
                        keys.append(i)

                dad = data.copy()  # copiem el dataset per no perdre les que aniran soles
                dad = dad.drop(columns=keys)  # eliminem les que van soles

                # windowing de tot el grup
                dad = self.windowing_group(dad, aux[0], aux[1])

                # afegim les que haviem tret abans
                for i in keys:
                    dad[i] = data[i]

            else:
                # cas de que no tinguem grups, son tots individuals
                dad = data.copy()  # copiem el dataset
                keys = list()
                for i in look_back.keys():
                    if i != -1:
                        keys.append(i)

            # windowing de la resta que es fan 1 a 1
            variables = [col for col in data.columns if col not in keys]
            for i in variables:
                if i.startswith('timestamp'): continue
                aux = look_back[-1]
                dad = self.windowing_univariant(dad, aux[0], aux[1], i)
        else:
            # cas de no tenir windowing
            dad = data.copy()

        return dad

    @staticmethod
    def timestamp_to_attrs(dad, extra_vars):
        """
        Afageix columnes derivades de l'índex temporal al DataFreame 'dad' segons les opcions indicades en 'extra_vars'. \n'
        :param dad: Dataframe amb un índex timestamp
        :type dad: pd.DataFrame
        :param extra_vars: Diccionari amb opcions per a generar columnes adicionals ('variables', 'festius')
        :type extra_vars: dict
        :return: El mateix DataFrame amb les noves columnes afegides.
        """

        if 'timestamp' in dad.columns:
            dad.index = pd.to_datetime(dad['timestamp'])

        if not extra_vars:
            # si extra_vars és None o buit, no cal fer res
            return dad

        # afegim columnes basades en l'índex temporal
        if 'variables' in extra_vars:
            for var in extra_vars['variables']:
                if var == 'Dia':
                    dad['Dia'] = dad.index.dayofweek  # Dia de la setmana (0 = Dll, 6 = Dg)
                elif var == 'Hora':
                    dad['Hora'] = dad.index.hour  # Hora del dia
                elif var == 'Mes':
                    dad['Mes'] = dad.index.month  # Mes de l'any
                elif var == 'Minut':
                    dad['Minut'] = dad.index.minute  # Minut de l'hora

        # Afegim columnes per a dies festius
        if 'festius' in extra_vars:
            festius = extra_vars['festius']
            if len(festius) == 1:
                # Festius d'un sol país
                h = holidays.country_holidays(festius[0])
            elif len(festius) == 2:
                # festius d'un sol país amb una regió específica
                h = holidays.country_holidays(festius[0], festius[1])
            else:
                raise ValueError("La clau 'festius' només suporta 1 o 2 paràmetres (país i opcionalment regió)")

            # Afageix una columna booleana indicant si cada dia es festiu
            dad['festius'] = dad.index.strftime('%Y-%m-%d').isin(h)

        if 'timestamp' in dad.columns:
            dad.drop(columns=['timestamp'], inplace=True)

        return dad

    @staticmethod
    def colinearity_remove(data, y, level=0.9):
        """
        Elimina les colinearitats entre les variables segons el nivell indicat
        :param data: Dataframe amb datetime com a índex
        :type data: pd.DataFrame
        :param y: Variable objectiu (per mirar que no la eliminem!)
        :param level: el percentatge de correlació de pearson per eliminar variables. None per no fer res
        :return:
            - dataset - Dataset amb les variables eliminades
            - to_drop - Les variables que ha eliminat
        """

        if level is None:
            dataset = data
            to_drop = None
        else:
            # eliminem els atributs massa correlacionats
            corr_matrix = data.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [column for column in upper.columns if any(upper[column] > level)]
            if np.array(to_drop == y).any():
                del to_drop[to_drop == y]
            data.drop(to_drop, axis=1, inplace=True)
            dataset = data.copy()

        return [dataset, to_drop]

    @staticmethod
    def scalate_data(data, input_scaler=None):
        """
        Escala les dades del dataset
        :param data: Dataset a escalar
        :param scaler: Mètode a usar per escalar
        :return: [Dataset , scaler]
        """
        dad = data.copy()
        scaler = None

        if input_scaler is not None:
            if input_scaler == 'minmax':
                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaler.fit(data)
                dad = scaler.transform(dad)
            elif input_scaler == 'robust':
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                scaler.fit(data)
                dad = scaler.transform(data)
            elif input_scaler == 'standard':
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                scaler.fit(data)
                dad = scaler.transform(dad)
            else:
                raise ValueError('Atribut Scaler no definit')
        else:
            scaler = None

        return dad, scaler

    @staticmethod
    def get_attribs(X, y, method=None):
        """
        Fa una selecció d'atributs
        :param X: Array amb les dades
        :param y: Array amb les dades
        :param method:
            - None: No fa res
            - integer: selecciona el número de features que s'indiquin
            - PCA: aplica un PCA per comprimir el dataset
        :return:
        """

        if method is None:
            model_select = []
            X_new = X
        elif method == 'Tree':
            from sklearn.ensemble import ExtraTreesRegressor
            from sklearn.feature_selection import SelectFromModel

            clf = ExtraTreesRegressor(n_estimators=50)
            clf = clf.fit(X, y)
            model_select = SelectFromModel(clf, prefit=True)
            X_new = model_select.transform(X)
        elif type(method) == int:
            from sklearn.feature_selection import SelectKBest, f_classif
            model_select = SelectKBest(f_classif, k=method)
            X_new = model_select.fit_transform(X, y)
        else:
            raise ValueError('Atribut de mètode de selecció no definit')

        return [model_select, X_new, y]

    def Model(self, X, y, algorithm=None, params=None, max_time=None):
        """
        Realitza un randomized search per trovar una bona configuració de l'algorisme indicat, o directament, es crea amb els paràmetres indicats.
        :param X:
        :param y:
        :param algorithm:
        :param params:
        :param max_time:
        :return:
        """

        import json

        with open(self.search_space_config_file) as json_file:
            d = json.load(json_file)

        if type(params) == dict:
            try:
                impo1 = d[algorithm][2]
                impo2 = d[algorithm][3]
            except:
                raise ValueError("Undefined Firecasting Algorithm!")

            a = __import__(impo1, globals(), locals(), [impo2])
            forecast_algorithm = eval("a. " + impo2)

            f = forecast_algorithm()
            f.set_params(**params)
            f.fit(X, y)
            score = 'none'
            return [f, score]
        elif params is None:  # no tenim paràmetres. Els busquem
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

            from sklearn.model_selection import ParameterSampler
            from sklearn.metrics import mean_absolute_error
            import time

            best_mae = np.inf

            # preparem la llista d'algorismes que volem provar
            if algorithm is None:
                # si no ens passen res els probem tots
                algorithm_list = list(d.keys())
            elif isinstance(algorithm, list):
                # passen una llista a probar
                algorithm_list = algorithm
            else:
                # ens passen només 1
                algorithm_list = [algorithm]

            best_overall_model = None
            best_overall_score = float("inf")
            # per cada algorisme a probar....
            for i in range(len(algorithm_list)):
                random_grid = d[algorithm_list[i]][0]

                if max_time is None:
                    iters = d[algorithm_list[i]][1]
                else:
                    iters = max_time

                impo1 = d[algorithm_list[i]][2]
                impo2 = d[algorithm_list[i]][3]

                if self.debug:
                    logger.info("  ")
                    logger.info(
                        f"Començant a optimitzar:  {algorithm_list[i]}  - Algorisme {str(algorithm_list.index(algorithm_list[i]) + 1)}  de  {str(len(algorithm_list))} - Maxim comput aprox (segons): {str(iters)}")

                sampler = ParameterSampler(random_grid, n_iter=np.iinfo(np.int64).max)

                a = __import__(impo1, globals(), locals(),
                               [impo2])  # equivalent a (import sklearn.svm as a) dinamicament
                forecast_algorithm = eval("a. " + impo2)

                try:
                    # creem i evaluem els models 1 a 1
                    t = time.perf_counter()

                    if self.debug:
                        logger.debug(f"Maxim {str(len(sampler))}  configuracions a probar!")
                        j = 0

                    best_model = None
                    best_mae = float("inf")

                    for params in sampler:
                        try:
                            regr = forecast_algorithm(**params)
                        except Exception as e:
                            logger.warning(f"Failed with params: {params}")
                            continue

                        pred_test = regr.fit(X_train, y_train).predict(X_test)
                        act = mean_absolute_error(y_test, pred_test)
                        if best_mae > act:
                            best_model = regr
                            best_mae = act

                        if self.debug:
                            logger.info("\r")
                            j += 1
                            logger.debug(
                                f"{str(j)} / {str(len(sampler))} Best MAE: {str(best_mae)} ({type(best_model).__name__}) Last MAE: {str(act)}  Elapsed: {str(time.perf_counter() - t)} s ")

                        if (time.perf_counter() - t) > iters:
                            if self.debug:
                                logger.debug("Algorisme " + algorithm_list[
                                    i] + " -- Abortat per Maxim comput aprox (segons): " + str(iters))
                                break

                except Exception as e:
                    logger.warning(f"WARNING: Algorisme {algorithm_list[i]},  -- Abortat per Motiu: {str(e)}")
                    continue

                if best_model is not None:
                    best_model.fit(X, y)
                    if best_mae < best_overall_score:
                        best_overall_model = best_model
                        best_overall_score = best_mae

            if best_overall_model is None:
                logger.error("❌ [DEBUG] No suitable model found!")
            else:
                 logger.info(f"✅ [DEBUG] Best Algorithm Found: {type(best_overall_model).__name__}")
                 logger.info(f"✅ [DEBUG] Best Score (MAE): {best_overall_score}")
                 # Try to log params if available
                 if hasattr(best_overall_model, 'get_params'):
                      logger.info(f"✅ [DEBUG] Best Params: {best_overall_model.get_params()}")

            return [best_overall_model, best_overall_score]
        else:
            raise ValueError('Paràmetres en format incorrecte!')

    @staticmethod
    def prepare_dataframes(sensor, meteo, extra_sensors):
        """
        Prepara els df juntant-los en un de sol amb millor gestió de NaNs
        """
        # Normalitza timestamps del sensor principal
        sensor = sensor.copy()
        sensor['timestamp'] = pd.to_datetime(sensor['timestamp']).dt.tz_localize(None).dt.floor('h')
        sensor = sensor.drop_duplicates(subset=['timestamp'])
        sensor = sensor.sort_values('timestamp')
        
        # CRÍTIC: Crea un rang temporal complet (sense gaps)
        date_range = pd.date_range(
            start=sensor['timestamp'].min(),
            end=sensor['timestamp'].max(),
            freq='h'
        )
        
        # Crea un DataFrame base amb totes les hores
        base_df = pd.DataFrame({'timestamp': date_range})
        
        # Merge amb el sensor (això omplirà gaps amb NaN que després interpolarem)
        merged_data = pd.merge(base_df, sensor, on='timestamp', how='left')
        
        # Interpola valors del sensor principal ABANS de fer altres merges
        if 'value' in merged_data.columns:
            nan_count = merged_data['value'].isna().sum()
            if nan_count > 0:
                logger.info(f"   📊 Interpolant {nan_count} NaNs del sensor principal")
                merged_data['value'] = merged_data['value'].interpolate(method='linear', limit_direction='both')
                # Si encara queden NaNs, emplena amb forward/backward fill
                merged_data['value'] = merged_data['value'].ffill().bfill()
        
        # Afegeix meteo
        if meteo is not None:
            meteo = meteo.copy()
            meteo['timestamp'] = pd.to_datetime(meteo['timestamp']).dt.tz_localize(None).dt.floor('h')
            meteo = meteo.drop_duplicates(subset=['timestamp'])
            
            # LEFT join per mantenir totes les files del sensor
            merged_data = pd.merge(merged_data, meteo, on='timestamp', how='left')
            
            # Interpola dades meteo
            meteo_cols = [col for col in meteo.columns if col != 'timestamp']
            for col in meteo_cols:
                if col in merged_data.columns:
                    nan_count = merged_data[col].isna().sum()
                    if nan_count > 0:
                        logger.info(f"   🌤️ Interpolant {nan_count} NaNs de {col}")
                        merged_data[col] = merged_data[col].interpolate(method='linear', limit_direction='both')
                        merged_data[col] = merged_data[col].ffill().bfill()
        
        # Afegeix sensors extra
        if extra_sensors is not None and len(extra_sensors) > 0:
            for sensor_name, sensor_df in extra_sensors.items():
                sensor_df = sensor_df.copy()
                sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp']).dt.tz_localize(None).dt.floor('h')
                sensor_df = sensor_df.drop_duplicates(subset=['timestamp'])
                
                # Renombra columnes per evitar conflictes (si té 'value')
                if 'value' in sensor_df.columns:
                    sensor_df = sensor_df.rename(columns={'value': f'value_{sensor_name}'})
                    value_col = f'value_{sensor_name}'
                else:
                    value_col = None
                
                merged_data = pd.merge(merged_data, sensor_df, on='timestamp', how='left')
                
                # Interpola el sensor extra
                if value_col and value_col in merged_data.columns:
                    nan_count = merged_data[value_col].isna().sum()
                    if nan_count > 0:
                        logger.info(f"   🔧 Interpolant {nan_count} NaNs de {sensor_name}")
                        merged_data[value_col] = merged_data[value_col].interpolate(method='linear', limit_direction='both')
                        merged_data[value_col] = merged_data[value_col].ffill().bfill()
        
        # Verifica NaNs finals
        total_nans = merged_data.isna().sum().sum()
        if total_nans > 0:
            logger.warning(f"⚠️ Encara queden {total_nans} NaNs després de preprocessing")
            for col in merged_data.columns:
                nan_count = merged_data[col].isna().sum()
                if nan_count > 0:
                    logger.warning(f"      {col}: {nan_count} NaNs ({nan_count/len(merged_data)*100:.1f}%)")
        else:
            logger.info(f"✅ Cap NaN després de preprocessing!")
        
        return merged_data

    def create_model(self, data, sensors_id, y, lat, lon, algorithm=None, params=None, escalat=None,
                     max_time=None, filename='newModel', meteo_data: pd.DataFrame = None, extra_sensors_df=None,
                     start_date=None, end_date=None):
        """
        Funció per crear, guardar i configurar el model de forecasting.

        :param lon: Longitud de la ubicació
        :param lat: Latitud de la ubicació
        :param extra_sensors_df: Sensors addicionals per al model
        :param data: Dataframe amb datetime com a índex
        :param sensors_id: ID del sensor objectiu
        :param y: Nom de la columna amb la variable a predir
        :param filename: Nom del fitxer per guardar el model
        :param max_time: Temps màxim d'entrenament
        :param escalat: Mètode d'escalat de dades
        :param params: Paràmetres del model
        :param algorithm: Algorisme a utilitzar
        :param meteo_data: Dades meteorològiques de la data
        :param start_date: Data d'inici per filtrar les dades (format: 'YYYY-MM-DD' o datetime). Si és None, s'usen totes les dades.
        :param end_date: Data de fi per filtrar les dades (format: 'YYYY-MM-DD' o datetime). Si és None, s'usen totes les dades.
        """

        extra_vars = {'variables': ['Dia', 'Hora', 'Mes'], 'festius': ['ES', 'CT']}
        feature_selection = 'Tree'
        colinearity_remove_level = 0.9
        look_back = {-1: [25, 48]}

        logger.info("*****************************************************************")
        logger.info(f"🔎 [DEBUG] Starting create_model")
        logger.info(f"   - Sensors ID: {sensors_id}")
        logger.info(f"   - Target Variable (y): {y}")
        logger.info(f"   - Input Data Shape: {data.shape}")

        # Filtrar dades per rang de dates si s'especifica
        if start_date is not None or end_date is not None:
            if 'timestamp' not in data.columns:
                logger.warning("⚠️ No es pot filtrar per dates: la columna 'timestamp' no existeix")
            else:
                data_copy = data.copy()
                data_copy['timestamp'] = pd.to_datetime(data_copy['timestamp'])
                # Mirem que les dades estiguin en format tz per a la comparació 
                if data_copy['timestamp'].dt.tz is not None:
                    data_copy['timestamp'] = data_copy['timestamp'].dt.tz_localize(None)
                
                if start_date is not None:
                    start_date_dt = pd.to_datetime(start_date)
                    data_copy = data_copy[data_copy['timestamp'] >= start_date_dt]
                    logger.info(f"📅 Filtrant dades des de: {start_date_dt.strftime('%Y-%m-%d')}")
                
                if end_date is not None:
                    end_date_dt = pd.to_datetime(end_date)
                    data_copy = data_copy[data_copy['timestamp'] <= end_date_dt]
                    logger.info(f"📅 Filtrant dades fins a: {end_date_dt.strftime('%Y-%m-%d')}")
                
                if data_copy.empty:
                    logger.error(f"❌ No hi ha dades en el rang especificat ({start_date} - {end_date})")
                    return

                # Aconseguim les dades filtrades i el sistema ens dirà quantes estem fent servir en total
                data = data_copy
                logger.info(f"✅ Utilitzant {len(data)} registres després del filtratge")

        # Descarregar dades meteo si no es proporcionen
        if meteo_data is not None and not data.empty:
            start_date = data['timestamp'].min().strftime("%Y-%m-%d")
            end_date = data['timestamp'].max().strftime("%Y-%m-%d")

            logger.info(f"⛅ Descarregant dades meteo històriques de {start_date} a {end_date}")

            url = (
                f"https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}"
                f"&start_date={start_date}&end_date={end_date}"
                f"&hourly=temperature_2m,relativehumidity_2m,dewpoint_2m,apparent_temperature,"
                f"precipitation,rain,weathercode,pressure_msl,surface_pressure,cloudcover,"
                f"cloudcover_low,cloudcover_mid,cloudcover_high,et0_fao_evapotranspiration,"
                f"vapor_pressure_deficit,windspeed_10m,windspeed_100m,winddirection_10m,"
                f"winddirection_100m,windgusts_10m,shortwave_radiation,direct_radiation,"
                f"diffuse_radiation,direct_normal_irradiance,terrestrial_radiation"
            )

            try:
                response = requests.get(url).json()
                hourly = response.get("hourly", {})
                timestamps = pd.to_datetime(hourly["time"])
                meteo_data = pd.DataFrame(hourly)
                meteo_data["timestamp"] = timestamps
                meteo_data.drop(columns=["time"], inplace=True)
            except Exception as e:
                logger.error(f"❌ No s'han pogut descarregar les dades meteo històriques: {e}")
                meteo_data = None

        # PREP PAS 0 - preparar els df de meteo-data i dades extra
        merged_data = self.prepare_dataframes(data, meteo_data, extra_sensors_df)
        
        logger.info(f"🔎 [DEBUG] After prepare_dataframes:")
        logger.info(f"   - Merged Data Shape: {merged_data.shape}")
        logger.info(f"   - NaNs totals: {merged_data.isna().sum().sum()}")

        if merged_data.empty:
            logger.error(f"\n ************* \n ❌ No hi ha dades per a realitzar el Forecast \n *************")
            return

        # PAS 1 - Fer el Windowing
        dad = self.do_windowing(merged_data, look_back)
        logger.info(f"🔎 [DEBUG] After Windowing:")
        logger.info(f"   - Shape: {dad.shape}")
        logger.info(f"   - NaNs totals: {dad.isna().sum().sum()}")

        # ELIMINA FILES AMB MASSA NaNs
        
        nan_threshold = 0.3  # Si més del 30% són NaNs, elimina la fila
        before_rows = len(dad)
        
        # Calcula percentatge de NaNs per fila
        nan_per_row = dad.isna().sum(axis=1) / len(dad.columns)
        dad = dad[nan_per_row <= nan_threshold]
        
        after_rows = len(dad)
        eliminated = before_rows - after_rows
        
        if eliminated > 0:
            logger.warning(f"⚠️ Eliminades {eliminated} files amb >{nan_threshold*100}% NaNs")
        else:
            logger.info(f"✅ Cap fila eliminada (totes tenen <{nan_threshold*100}% NaNs)")
        
        # Després, omple els NaNs restants amb bfill/ffill
        nans_before = dad.isna().sum().sum()
        logger.info(f"   NaNs abans bfill: {nans_before}")
        
        dad = dad.bfill().ffill()
        
        nans_after = dad.isna().sum().sum()
        logger.info(f"   NaNs després bfill: {nans_after}")
        
        if nans_after > 0:
            logger.error(f"❌ ALERTA: Encara queden {nans_after} NaNs després de bfill/ffill!")
            logger.error("   Columnes amb NaNs:")
            for col in dad.columns:
                nan_count = dad[col].isna().sum()
                if nan_count > 0:
                    logger.error(f"      {col}: {nan_count}")
        # ============================================================

        # PAS 2 - Crear variable dia_setmana, hora, més i meteoData
        dad = self.timestamp_to_attrs(dad, extra_vars)
        logger.info(f"🔎 [DEBUG] After Timestamp Attributes:")
        logger.info(f"   - Shape: {dad.shape}")

        # PAS 3 - Treure Col·linearitats
        [dad, to_drop] = self.colinearity_remove(dad, y, level=colinearity_remove_level)
        colinearity_remove_level_to_drop = to_drop
        logger.info(f"🔎 [DEBUG] After Colinearity Removal:")
        logger.info(f"   - Dropped Columns: {to_drop}")
        logger.info(f"   - Shape: {dad.shape}")

        # PAS 4 - Treure NaN
        dad.replace([np.inf, -np.inf], np.nan, inplace=True)
        X = dad.bfill().ffill()
        X = X.dropna()
        
        logger.info(f"🔎 [DEBUG] After final dropna:")
        logger.info(f"   - Shape: {X.shape}")
        logger.info(f"   - Files eliminades per NaN: {len(dad) - len(X)}")
        
        # PAS 5 - Desfer el dataset i guardar matrius X i y
        nomy = y
        y = pd.to_numeric(X[nomy], errors='raise')
        del X[nomy]

        # PAS 6 - Escalat
        X, scaler = self.scalate_data(X, escalat)

        # PAS 7 - Seleccionar atributs
        [model_select, X_new, y_new] = self.get_attribs(X, y, feature_selection)

        # PAS 8 - Crear el model
        [model, score] = self.Model(X_new, y_new.values, algorithm, params, max_time=max_time)

        # PAS 9 - Guardar el model
        if algorithm is None:
            self.db['max_time'] = max_time
            self.db['algorithm'] = "AUTO"
        else:
            self.db['params'] = params
            self.db['algorithm'] = algorithm

        if meteo_data is not None:
            self.db['meteo_data'] = meteo_data
            self.db['meteo_data_is_selected'] = True
        else:
            self.db['meteo_data'] = None
            self.db['meteo_data_is_selected'] = False

        self.db['model'] = model
        self.db['scaler'] = scaler
        self.db['scaler_name'] = escalat
        self.db['model_select'] = model_select
        self.db['colinearity_remove_level_to_drop'] = colinearity_remove_level_to_drop
        self.db['extra_vars'] = extra_vars
        self.db['look_back'] = look_back
        self.db['score'] = score
        self.db['objective'] = nomy
        self.db['initial_data'] = data
        self.db['sensors_id'] = sensors_id
        self.db['extra_sensors'] = extra_sensors_df
        self.db['lat'] = lat
        self.db['lon'] = lon

        self.save_model(filename)

        if self.debug:
            logger.debug('Model guardat! Score: ' + str(score))

    def forecast(self, data, y, model, future_steps=48):
        """
        Forecast autoregressiu: cada predicció alimenta el següent pas.
        """
    
        # Recuperem configuració del model
        model_select = self.db.get('model_select', [])
        scaler = self.db.get('scaler')
        colinearity_remove_level_to_drop = self.db.get('colinearity_remove_level_to_drop', [])
        extra_vars = self.db.get('extra_vars', {})
        look_back = self.db.get('look_back', {-1: [25, 48]})
    
        # Guardem valors reals (per a la part històrica del plot)
        if y not in data.columns:
            raise ValueError(f"Columna {y} no trobada en el dataset")
    
        real_values_column = data[y].copy()
    
        # Historial viu que anirem ampliant
        history = data.copy()
        predictions = []
    
        # ============================================================
        # FORECAST AUTOREGRESSIU PAS A PAS
        # ============================================================
        for step in range(future_steps):
    
            # PAS 1 - Windowing
            df_step = self.do_windowing(history, look_back)
    
            # PAS 1.1 - Neteja de NaNs
            nan_threshold = 0.3
            nan_per_row = df_step.isna().sum(axis=1) / len(df_step.columns)
            df_step = df_step[nan_per_row <= nan_threshold]
            df_step = df_step.bfill().ffill()
    
            # PAS 2 - Variables temporals
            df_step = self.timestamp_to_attrs(df_step, extra_vars)
    
            # PAS 3 - Eliminar colinearitats
            if colinearity_remove_level_to_drop:
                cols_to_drop = [
                    col for col in colinearity_remove_level_to_drop
                    if col in df_step.columns
                ]
                df_step.drop(cols_to_drop, axis=1, inplace=True)
    
            # PAS 4 - Separar X / y
            X_step = df_step.drop(columns=[y])
            X_step = X_step.iloc[[-1]]
    
            # PAS 5 - Escalat
            if scaler:
                X_step = pd.DataFrame(
                    scaler.transform(X_step),
                    index=X_step.index,
                    columns=X_step.columns
                )
    
            # PAS 6 - Selecció de features
            if model_select:
                X_step = model_select.transform(X_step)
    
            # PAS 7 - Predicció
            y_pred = float(model.predict(X_step)[0])
            predictions.append(y_pred)
    
            # PAS 8 - Afegir a l'historial
            next_timestamp = history.index[-1] + pd.Timedelta(hours=1)
    
            new_row = history.iloc[[-1]].copy()
            new_row.index = [next_timestamp]
            new_row[y] = y_pred
    
            history = pd.concat([history, new_row])
    
        # ============================================================
        # Construcció del DataFrame final
        # ============================================================
        future_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(hours=1),
            periods=future_steps,
            freq='H'
        )
    
        forecast_df = pd.DataFrame(
            predictions,
            index=future_index,
            columns=[y]
        )
    
        return forecast_df, real_values_column, self.db['sensors_id']

    def save_model(self, model_filename):
        """
        Guarda el model en un arxiu .pkl i l'elimina de la base de daades interna del forecast (self.db)
        :param model_filename: Nom que es vol donar al fitxer, si és nul serà "savedModel"
        """
        full_path = os.path.join(self.models_filepath,'forecastings/', model_filename + ".pkl")
        os.makedirs(self.models_filepath + 'forecastings', exist_ok=True)

        if os.path.exists(full_path):
            logger.warning(f"El fitxer {full_path} ja existeix. S'eliminarà.")
            os.remove(full_path)

        joblib.dump(self.db, full_path)
        logger.info(f"  💾 Model guardat al fitxer {model_filename}.pkl")

        self.db.clear()

    def load_model(self, model_filename):
        self.db = joblib.load(self.models_filepath + 'forecastings/' + model_filename)
        logger.info(f"  💾 Model carregat del fitxer {model_filename}")
