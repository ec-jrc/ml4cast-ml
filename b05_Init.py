import src.constants as cst
import os

def init(aoi):
    '''Init directories and file names'''
    return {
        'Algeria': {
            "AOI": aoi,
            "prct2retain": 90,  # percent of production to retain
            "timeRange": range(2002, 2019), #2002 because data re from 2001 10 01
            'crop_IDs': [1, 2, 3],
            # Input/Output specs
            "input_dir": cst.idir,                                        #local_base_dir + '/ML1_data_input'
            "output_dir": os.path.join(cst.odir, aoi), # 'ML1_data_output',
            # Definition of input variables (names must match the filename from ASAP
            "ivars": ['NDVI', 'rad', 'rainfall', 'temperature'],
            "ivars_short": ['ND', 'Rad', 'Rain', 'T'],
            "iunits": ['-', 'kJ m-2 dek-1', 'mm', '°C'],
            #Pheno parameters
            'prctSOS': 20,
            'prctSEN' :70,
            'prctEOS' : 20
        },
        'Mali': {
            "AOI": aoi,
            "timeRange": range(2002, 2020),  # 2003 because data re from 2001 10 01
            # Input/Output specs
            "input_dir": cst.idir,
            "output_dir": os.path.join(cst.odir, aoi),     #'ML1_data_output',
            # Definition of input variables (names must match the filename from ASAP
            "ivars": ['NDVI', 'rad', 'rainfall', 'temperature'],
            "ivars_short": ['ND', 'Rad', 'Rain', 'T'],
            "iunits": ['-', 'kJ m-2 dek-1', 'mm', '°C'],
            # Pheno parameters
            'prctSOS': 20,
            'prctSEN': 70,
            'prctEOS': 20
        },
        'BF': {
            "AOI": aoi,
            "timeRange": range(2002, 2020),  # 2003 because data re from 2001 10 01
            # Input/Output specs
            "input_dir": cst.idir,
            "output_dir": os.path.join(cst.odir, aoi),
            # Definition of input variables (names must match the filename from ASAP
            "ivars": ['NDVI', 'rad', 'rainfall', 'temperature'],
            "ivars_short": ['ND', 'Rad', 'Rain', 'T'],
            "iunits": ['-', 'kJ m-2 dek-1', 'mm', '°C'],
            # Pheno parameters
            'prctSOS': 20,
            'prctSEN': 70,
            'prctEOS': 20
        },
        'ZAsummer': {
            "AOI": aoi,
            "prct2retain": 100, # percent of production to retain, used to exclude marginal regions
            "timeRange": range(2002, 2021),  # in year of EOS summer crops starting in dec
            'crop_IDs': [1, 2, 3],
            # Input/Output specs
            "input_dir": cst.idir,
            "output_dir": os.path.join(cst.odir, aoi),
            # Definition of input variables (names must match the filename from ASAP
            # "ivars": ['NDVI', 'rad', 'rainfall', 'temperature'],
            "ivars": ['NDVI', 'Rad', 'rainfall', 'temperature', 'soil_moisture'],
            "ivars_short": ['ND', 'Rad', 'Rain', 'T', 'SM'],
            "iunits": ['-', 'kJ m-2 dek-1', 'mm', '°C', 'm3/m3'],
            # "ivars_short": ['ND', 'Rad', 'Rain', 'T'],
            # "iunits": ['-', 'kJ m-2 dek-1', 'mm', '°C'],
            # Pheno parameters
            'prctSOS': 20,
            'prctSEN': 70,
            'prctEOS': 20
        },
        'Other': {}
    }.get(aoi, 'Target not found')


