import os
import sys
import warnings
import numpy as np
import pandas as pd
import pytest

from tests.climate_setpoints import get_ncc_setpoints, apply_setpoints_to_building

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from pybuildingenergy.source.utils import ISO52016
from pybuildingenergy.source.check_input import sanitize_and_validate_BUI
from pybuildingenergy.source.DHW import Volume_and_energy_DHW_calculation, generate_calendar

 #  3) CONSTRUCTION U-VALUES & THERMAL CAPACITY

# U-values (W / m²·K) — Australian BCA 2006 minimum-spec
U_EXT_WALL  = 1.00   # brick veneer / precast w/ R1.0 insulation
U_INT_WALL  = 2.50   # concrete block + plasterboard, no insulation
U_INT_SLAB  = 1.80   # 200 mm concrete intermediate floor
U_WINDOW    = 5.40   # aluminium-frame single glazing
G_WINDOW    = 0.65   # SHGC of clear single glazing

# Solar absorptance — DARK RED BRICK (confirmed from photo)
ABS_EXT_WALL = 0.75  # dark red brick (was 0.55 for mid-tone, corrected after seeing photo)
ABS_INT      = 0.0   # interior surfaces

# Areal thermal capacity (J / m²·K)
C_EXT_WALL = 450_000   # heavy concrete external wall
C_INT_WALL = 330_000   # concrete-block partition
C_INT_SLAB = 480_000   # 200 mm concrete slab
C_WINDOW   = 0


#2 Geometry

LEN_NS  = 5.0   # N-S length, m  (width of west facade — along Barry St which runs N-S)
LEN_EW  = 4.0   # E-W depth,  m  (apartment depth — perpendicular into the building)
HEIGHT  = 2.7   # ceiling height, m

FLOOR_AREA = LEN_NS * LEN_EW                # 20.0 m²
VOLUME     = FLOOR_AREA * HEIGHT            # 54.0 m³

# Wall gross areas
A_WEST_GROSS  = LEN_NS * HEIGHT             # 13.5 m²  EXTERIOR — faces Barry St, has windows
A_EAST_GROSS  = LEN_NS * HEIGHT             # 13.5 m²  interior — to corridor
A_NORTH_GROSS = LEN_EW * HEIGHT             # 10.8 m²  interior — to Apt 306
A_SOUTH_GROSS = LEN_EW * HEIGHT             # 10.8 m²  interior — to Apt 304

# Two small horizontal-slider windows on the WEST wall (facing Barry St)
# From site photos: 2 windows side-by-side, each ~0.9 m wide × 0.9 m tall
# Only the right-hand (operable/slider) window opens; left is fixed.
WIN_WIDTH_FIXED,    WIN_HEIGHT_FIXED    = 0.9, 0.9   # fixed glazing
WIN_WIDTH_OPERABLE, WIN_HEIGHT_OPERABLE = 0.9, 0.9   # operable (horizontal slider)

A_WINDOW_FIXED    = WIN_WIDTH_FIXED * WIN_HEIGHT_FIXED         # 0.81 m²
A_WINDOW_OPERABLE = WIN_WIDTH_OPERABLE * WIN_HEIGHT_OPERABLE   # 0.81 m²
A_WINDOW_TOTAL    = A_WINDOW_FIXED + A_WINDOW_OPERABLE         # 1.62 m²

# Opaque part of the west wall (= gross − both windows)
A_WEST_OPAQUE = A_WEST_GROSS - A_WINDOW_TOTAL                  # 11.88 m²

# Sanity check: window-to-wall ratio of the west facade ≈ 12%
# 0.9 + 0.9 = 1.8 m of glazing across a 5 m wide facade = 36% width coverage
# Window heights are 0.9 m on a 2.7 m wall = 33% height coverage


@pytest.fixture
def building_data():
    _lat = -37.8136
    _lon = 144.9695
    _sp = get_ncc_setpoints(lat=_lat, lon=_lon)
    _single_cooling = (_sp["cooling_setpoint_bedroom"] + _sp["cooling_setpoint_living"]) / 2.0
    # _single_cooling = 40.0  # override with single cooling setpoint for simplicity in this test

    # return {
    #     "building": {
    #         "name": "ML_Target_Building_001",
    #         "azimuth_relative_to_true_north": 41.8,
    #         "latitude": _lat,
    #         "longitude": _lon,
    #         "exposed_perimeter": 40,
    #         "height": 3,
    #         "wall_thickness": 0.3,
    #         "n_floors": 1,
    #         "building_type_class": "Residential_apartment",
    #         "adj_zones_present": False,
    #         "number_adj_zone": 2,
    #         "net_floor_area": 100,
    #         "construction_class": "class_i"
    #     },
    #     "adjacent_zones": [
    #         {
    #             "name": "adj_1",
    #             "orientation_zone": {"azimuth": 0},
    #             "area_facade_elements": np.array([20, 60, 30, 30, 50, 50], dtype=object),
    #             "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
    #             "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
    #             "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
    #             'volume': 300,
    #             'building_type_class': 'Residential_apartment',
    #             'a_use': 50
    #         },
    #         {
    #             "name": "adj_2",
    #             "orientation_zone": {"azimuth": 180},
    #             "area_facade_elements": np.array([20, 60, 30, 30, 50, 50], dtype=object),
    #             "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
    #             "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
    #             "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
    #             'volume': 300,
    #             'building_type_class': 'Residential_apartment',
    #             'a_use': 50
    #         }
    #     ],
    #     "building_surface": [
    #         {
    #             "name": "Roof surface",
    #             "type": "opaque",
    #             "area": 130,
    #             "sky_view_factor": 1.0,
    #             "u_value": 2.2,
    #             "solar_absorptance": 0.4,
    #             "thermal_capacity": 741500.0,
    #             "orientation": {"azimuth": 0, "tilt": 0},
    #             "name_adj_zone": None
    #         },
    #         {
    #             "name": "Opaque north surface",
    #             "type": "opaque",
    #             "area": 30,
    #             "sky_view_factor": 0.0,
    #             "basement_depth": 2.5,
    #             "u_value": 1.4,
    #             "solar_absorptance": 0.4,
    #             "thermal_capacity": 1416240.0,
    #             "orientation": {"azimuth": 0, "tilt": 90},
    #             "name_adj_zone": "adj_1"
    #         },
    #         {
    #             "name": "Opaque south surface",
    #             "type": "opaque",
    #             "area": 30,
    #             "sky_view_factor": 0.5,
    #             "u_value": 1.4,
    #             "solar_absorptance": 0.4,
    #             "thermal_capacity": 1416240.0,
    #             "orientation": {"azimuth": 180, "tilt": 90},
    #             "name_adj_zone": "adj_2"
    #         },
    #         {
    #             "name": "Opaque east surface",
    #             "type": "opaque",
    #             "area": 30,
    #             "sky_view_factor": 0.5,
    #             "u_value": 1.2,
    #             "solar_absorptance": 0.6,
    #             "thermal_capacity": 1416240.0,
    #             "orientation": {"azimuth": 90, "tilt": 90},
    #             "name_adj_zone": None
    #         },
    #         {
    #             "name": "Opaque west surface",
    #             "type": "opaque",
    #             "area": 30,
    #             "sky_view_factor": 0.5,
    #             "u_value": 1.2,
    #             "solar_absorptance": 0.7,
    #             "thermal_capacity": 1416240.0,
    #             "orientation": {"azimuth": 270, "tilt": 90},
    #             "name_adj_zone": None
    #         },
    #         {
    #             "name": "Slab to ground",
    #             "type": "opaque",
    #             "area": 100,
    #             "sky_view_factor": 0.0,
    #             "u_value": 1.6,
    #             "solar_absorptance": 0.6,
    #             "thermal_capacity": 405801,
    #             "orientation": {"azimuth": 0, "tilt": 0},
    #             "name_adj_zone": None
    #         },
    #         {
    #             "name": "Transparent east surface",
    #             "type": "transparent",
    #             "area": 25,
    #             "sky_view_factor": 0.5,
    #             "u_value": 5,
    #             "g_value": 0.726,
    #             "height": 2,
    #             "width": 1,
    #             "parapet": 1.1,
    #             "orientation": {"azimuth": 90, "tilt": 90},
    #             "shading": False,
    #             "shading_type": "horizontal_overhang",
    #             "width_or_distance_of_shading_elements": 0.5,
    #             "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
    #             "name_adj_zone": None
    #         },
    #         {
    #             "name": "Transparent west surface",
    #             "type": "transparent",
    #             "area": 25,
    #             "sky_view_factor": 0.5,
    #             "u_value": 5,
    #             "g_value": 0.726,
    #             "height": 2,
    #             "width": 1,
    #             "parapet": 1.1,
    #             "orientation": {"azimuth": 270, "tilt": 90},
    #             "shading": False,
    #             "shading_type": "horizontal_overhang",
    #             "width_or_distance_of_shading_elements": 0.5,
    #             "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
    #             "name_adj_zone": None
    #         }
    #     ],
    #     "units": {
    #         "area": "m²",
    #         "u_value": "W/m²K",
    #         "thermal_capacity": "J/kgK",
    #         "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
    #         "tilt": "degrees (0=horizontal, 90=vertical)",
    #         "internal_gain": "W/m²",
    #         "internal_gain_profile": "Normalized to 0-1",
    #         "HVAC_profile": "0: off, 1: on"
    #     },
    #     "building_parameters": {
    #         "temperature_setpoints": {
    #             "heating_setpoint": _sp["heating_setpoint"],
    #             "heating_setback":  _sp["heating_setback"],
    #             "cooling_setpoint": _single_cooling,
    #             "cooling_setback":  _sp["cooling_setback"],
    #             "ncc_zone":         _sp["ncc_zone"],
    #             "units": "°C"
    #         },
    #         "system_capacities": {
    #             "heating_capacity": 10000000.0,
    #             "cooling_capacity": 12000000.0,
    #             "units": "W"
    #         },
    #         "airflow_rates": {
    #             "infiltration_rate": 1.0,
    #             "units": "ACH (air changes per hour)"
    #         },
    #         "internal_gains": [
    #             {
    #                 "name": "occupants",
    #                 "full_load": 4.2,
    #                 "weekday": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 1.0, 1.0],
    #                 "weekend": [1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0]
    #             },
    #             {
    #                 "name": "appliances",
    #                 "full_load": 3,
    #                 "weekday": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
    #                 "weekend": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
    #             },
    #             {
    #                 "name": "lighting",
    #                 "full_load": 3,
    #                 "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
    #                 "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
    #             }
    #         ],
    #         "construction": {
    #             "wall_thickness": 0.3,
    #             "thermal_bridges": 2,
    #             "units": "m (for thickness), W/mK (for thermal bridges)"
    #         },
    #         "climate_parameters": {
    #             "coldest_month": 1,
    #             "units": "1-12 (January-December)"
    #         },
    #         "heating_profile": {
    #             "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    #             "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    #         },
    #         "cooling_profile": {
    #             "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
    #             "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    #         },
    #         "ventilation": {
    #             "ventilation_type": "custom",
    #             "flow_rate_per_person": 0.005,
    #             "custom_heat_transfer_coefficient_ventilation": 2.0,
    #             "weekday": [1.0] * 24,
    #             "weekend": [1.0] * 24
    #         },
    #         "ventilation_profile": {
    #             "weekday": [1.0] * 24,
    #             "weekend": [1.0] * 24
    #         }
    #     }
    # }


    return {
            "building": {
                "name": "Apt_305_50_Barry_St_Carlton",
                "azimuth_relative_to_true_north": 270,
                "latitude":  -37.800,
                "longitude": 144.968,
                "exposed_perimeter": 18,
                "height": HEIGHT,
                "wall_thickness": 0.20,
                "n_floors": 1,
                "building_type_class": "Residential_apartment",
                "adj_zones_present": True,
                "number_adj_zone": 5,
                "net_floor_area": FLOOR_AREA,
                "construction_class": "class_iii",
                "construction_year": "2006-today",
                "country": "Australia",
            },
            # --------- 4b) ADJACENT ZONES ---------------------------
            "adjacent_zones": [
                {   # Apt 405 — studio above
                    "name": "apt_above",
                    "orientation_zone": {"azimuth": 270.0},
                    "area_facade_elements":    np.array([A_WEST_GROSS, A_NORTH_GROSS, A_EAST_GROSS, A_SOUTH_GROSS, FLOOR_AREA, FLOOR_AREA]),
                    "typology_elements":       ["OP", "OP", "OP", "OP", "OP", "OP"],
                    "transmittance_U_elements":np.array([U_EXT_WALL, U_INT_WALL, U_INT_WALL, U_INT_WALL, U_INT_SLAB, U_INT_SLAB]),
                    "orientation_elements":    np.array(["WV", "NV", "EV", "SV", "HOR", "HOR"]),
                    "volume": VOLUME,
                    "building_type_class": "Residential_apartment",
                    "a_use": FLOOR_AREA,
                },
                {   # Apt 205 — studio below
                    "name": "apt_below",
                    "orientation_zone": {"azimuth": 270.0},
                    "area_facade_elements":    np.array([A_WEST_GROSS, A_NORTH_GROSS, A_EAST_GROSS, A_SOUTH_GROSS, FLOOR_AREA, FLOOR_AREA]),
                    "typology_elements":       ["OP", "OP", "OP", "OP", "OP", "OP"],
                    "transmittance_U_elements":np.array([U_EXT_WALL, U_INT_WALL, U_INT_WALL, U_INT_WALL, U_INT_SLAB, U_INT_SLAB]),
                    "orientation_elements":    np.array(["WV", "NV", "EV", "SV", "HOR", "HOR"]),
                    "volume": VOLUME,
                    "building_type_class": "Residential_apartment",
                    "a_use": FLOOR_AREA,
                },
                {   # Apt 306 — studio to the NORTH
                    "name": "apt_north",
                    "orientation_zone": {"azimuth": 0.0},
                    "area_facade_elements":    np.array([A_WEST_GROSS, A_NORTH_GROSS, A_EAST_GROSS, A_SOUTH_GROSS, FLOOR_AREA, FLOOR_AREA]),
                    "typology_elements":       ["OP", "OP", "OP", "OP", "OP", "OP"],
                    "transmittance_U_elements":np.array([U_INT_WALL, U_INT_WALL, U_INT_WALL, U_INT_WALL, U_INT_SLAB, U_INT_SLAB]),
                    "orientation_elements":    np.array(["WV", "NV", "EV", "SV", "HOR", "HOR"]),
                    "volume": VOLUME,
                    "building_type_class": "Residential_apartment",
                    "a_use": FLOOR_AREA,
                },
                {   # Apt 304 — studio to the SOUTH
                    "name": "apt_south",
                    "orientation_zone": {"azimuth": 180.0},
                    "area_facade_elements":    np.array([A_WEST_GROSS, A_NORTH_GROSS, A_EAST_GROSS, A_SOUTH_GROSS, FLOOR_AREA, FLOOR_AREA]),
                    "typology_elements":       ["OP", "OP", "OP", "OP", "OP", "OP"],
                    "transmittance_U_elements":np.array([U_INT_WALL, U_INT_WALL, U_INT_WALL, U_INT_WALL, U_INT_SLAB, U_INT_SLAB]),
                    "orientation_elements":    np.array(["WV", "NV", "EV", "SV", "HOR", "HOR"]),
                    "volume": VOLUME,
                    "building_type_class": "Residential_apartment",
                    "a_use": FLOOR_AREA,
                },
                {   # Building corridor — runs along the east side
                    "name": "corridor",
                    "orientation_zone": {"azimuth": 90.0},
                    "area_facade_elements":    np.array([81.0, 5.4, 81.0, 5.4, 60.0, 60.0]),
                    "typology_elements":       ["OP", "OP", "OP", "OP", "OP", "OP"],
                    "transmittance_U_elements":np.array([U_INT_WALL] * 6),
                    "orientation_elements":    np.array(["WV", "NV", "EV", "SV", "HOR", "HOR"]),
                    "volume": 162.0,
                    "building_type_class": "Residential_apartment",
                    "a_use": 60.0,
                },
            ],

            # --------- 4c) ENVELOPE SURFACES ------------------------
            "building_surface": [

                # 1) WEST EXTERIOR WALL — opaque brick (Barry St facade)
                {
                    "name": "West exterior wall (opaque)",
                    "type": "opaque",
                    "area": A_WEST_OPAQUE,
                    "sky_view_factor": 0.5,
                    "u_value": U_EXT_WALL,
                    "solar_absorptance": ABS_EXT_WALL,
                    "thermal_capacity": C_EXT_WALL,
                    "orientation": {"azimuth": 270.0, "tilt": 90.0},
                    "name_adj_zone": None,
                    "height": HEIGHT,
                    "length": LEN_NS,
                },

                # 2) NORTH INTERIOR WALL (to Apt 306)
                {
                    "name": "North wall to Apt 306",
                    "type": "opaque",
                    "area": A_NORTH_GROSS,
                    "sky_view_factor": 0.0,
                    "u_value": U_INT_WALL,
                    "solar_absorptance": ABS_INT,
                    "thermal_capacity": C_INT_WALL,
                    "orientation": {"azimuth": 0.0, "tilt": 90.0},
                    "name_adj_zone": "apt_north",
                    "height": HEIGHT,
                    "length": LEN_EW,
                },

                # 3) SOUTH INTERIOR WALL (to Apt 304)
                {
                    "name": "South wall to Apt 304",
                    "type": "opaque",
                    "area": A_SOUTH_GROSS,
                    "sky_view_factor": 0.0,
                    "u_value": U_INT_WALL,
                    "solar_absorptance": ABS_INT,
                    "thermal_capacity": C_INT_WALL,
                    "orientation": {"azimuth": 180.0, "tilt": 90.0},
                    "name_adj_zone": "apt_south",
                    "height": HEIGHT,
                    "length": LEN_EW,
                },

                # 4) EAST INTERIOR WALL (to corridor)
                {
                    "name": "East wall to corridor",
                    "type": "opaque",
                    "area": A_EAST_GROSS,
                    "sky_view_factor": 0.0,
                    "u_value": U_INT_WALL,
                    "solar_absorptance": ABS_INT,
                    "thermal_capacity": C_INT_WALL,
                    "orientation": {"azimuth": 90.0, "tilt": 90.0},
                    "name_adj_zone": "corridor",
                    "height": HEIGHT,
                    "length": LEN_NS,
                },

                # 5) FLOOR (to Apt 205 below)
                {
                    "name": "Floor to Apt 205",
                    "type": "opaque",
                    "area": FLOOR_AREA,
                    "sky_view_factor": 0.0,
                    "u_value": U_INT_SLAB,
                    "solar_absorptance": ABS_INT,
                    "thermal_capacity": C_INT_SLAB,
                    "orientation": {"azimuth": 0.0, "tilt": 0.0},
                    "name_adj_zone": "apt_below",
                    "height": LEN_NS,
                    "length": LEN_EW,
                },

                # 6) CEILING (to Apt 405 above)
                {
                    "name": "Ceiling to Apt 405",
                    "type": "opaque",
                    "area": FLOOR_AREA,
                    "sky_view_factor": 0.0,
                    "u_value": U_INT_SLAB,
                    "solar_absorptance": ABS_INT,
                    "thermal_capacity": C_INT_SLAB,
                    "orientation": {"azimuth": 0.0, "tilt": 0.0},
                    "name_adj_zone": "apt_above",
                    "height": LEN_NS,
                    "length": LEN_EW,
                },

                # 7) WEST WINDOW — FIXED (left-hand pane, non-opening)
                {
                    "name": "West window — fixed",
                    "type": "transparent",
                    "area": A_WINDOW_FIXED,
                    "sky_view_factor": 0.5,
                    "u_value": U_WINDOW,
                    "solar_absorptance": 0.5,
                    "thermal_capacity": C_WINDOW,
                    "orientation": {"azimuth": 270.0, "tilt": 90.0},
                    "name_adj_zone": None,
                    "height": WIN_HEIGHT_FIXED,
                    "g_value": G_WINDOW,
                    "width": WIN_WIDTH_FIXED,
                    "parapet": 1.0,
                    "shading": True,
                    "shading_type": "horizontal_overhang",
                    "width_or_distance_of_shading_elements": 0.05,
                    "overhang_proprieties": {
                        "width_of_horizontal_overhangs": 0.25,
                    },
                },

                # 8) WEST WINDOW — OPERABLE horizontal slider (right-hand pane)
                {
                    "name": "West window — operable",
                    "type": "transparent",
                    "area": A_WINDOW_OPERABLE,
                    "sky_view_factor": 0.5,
                    "u_value": U_WINDOW,
                    "solar_absorptance": 0.5,
                    "thermal_capacity": C_WINDOW,
                    "orientation": {"azimuth": 270.0, "tilt": 90.0},
                    "name_adj_zone": None,
                    "height": WIN_HEIGHT_OPERABLE,
                    "g_value": G_WINDOW,
                    "width": WIN_WIDTH_OPERABLE,
                    "parapet": 1.0,
                    "shading": True,
                    "shading_type": "horizontal_overhang",
                    "width_or_distance_of_shading_elements": 0.05,
                    "overhang_proprieties": {
                        "width_of_horizontal_overhangs": 0.25,
                    },
                },
            ],

            "units": {
                "area": "m²",
                "u_value": "W/m²K",
                "thermal_capacity": "J/kgK",
                "azimuth": "degrees (0=N, 90=E, 180=S, 270=W)",
                "tilt": "degrees (0=horizontal, 90=vertical)",
                "internal_gain": "W/m²",
                "internal_gain_profile": "Normalized to 0-1",
                "HVAC_profile": "0: off, 1: on"
            },
            "building_parameters": {
                "temperature_setpoints": {
                    "heating_setpoint": _sp["heating_setpoint"],
                    "heating_setback":  _sp["heating_setback"],
                    "cooling_setpoint": _single_cooling,
                    "cooling_setback":  _sp["cooling_setback"],
                    "ncc_zone":         _sp["ncc_zone"],
                    "units": "°C"
                },
                "system_capacities": {
                    "heating_capacity": 10000000.0,
                    "cooling_capacity": 12000000.0,
                    "units": "W"
                },
                "ventilation": {
                    "ventilation_type": "occupancy",
                    "flow_rate_per_person": 2.0,
                    "units": "l/(s m²)",
                    "custom_heat_transfer_coefficient_ventilation": None,
                    "info": "Annual-average rate; real summer ACH much higher (open window)",
                },

                "internal_gains": [
                    {
                        "name": "occupants",
                        "full_load": 8.0,    # 2 ppl × ~80 W metabolic / 20 m² = 8 W/m² peak
                        # Weekday: both home overnight, friend leaves 08, user mostly home,
                        # both back together 20:00 onward
                        #             0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23
                        "weekday": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.5,0.4,0.5,0.5,0.5,0.4,0.5,0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0],
                        # Weekend: friend home; user sometimes out — overall ~70-80 % occupied
                        "weekend": [1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.8,0.7,0.7,0.7,0.7,0.5,0.5,0.7,0.8,1.0,1.0,1.0,1.0,1.0,1.0],
                    },
                {
                        "name": "appliances",
                        "full_load": 25.0,   # ~500 W peak in 20 m² — driven by cooking spike
                        # Baseline = fridge (~50 W) + electronics when user works (~80 W).
                        # Spike at 20:00 = stove + air fryer + microwave simultaneously,
                        # no extractor → all heat retained. Small kettle bumps 07-08 & 22-23.
                        "weekday": [0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.3,0.2,0.2,0.2,0.2,0.3,0.2,0.2,0.2,0.2,0.3,0.3,0.4,1.0,0.6,0.4,0.2],
                        "weekend": [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,0.3,0.4,0.3,0.3,0.4,0.3,0.3,0.3,0.3,0.4,0.4,0.5,1.0,0.6,0.4,0.2],
                    },
                    {
                        "name": "lighting",
                        "full_load": 3.0,
                        "weekday": [0.0,0.0,0.0,0.0,0.0,0.0,0.3,0.3,0.1,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.5,0.8,0.8,0.8,0.7,0.4,0.1],
                        "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.3,0.3,0.2,0.2,0.2,0.2,0.2,0.2,0.3,0.5,0.8,0.8,0.8,0.7,0.4,0.1],
                    },
                ],

                "construction": {
                    "wall_thickness": 0.20,
                    "thermal_bridges": 1.5,
                    "units": "m (thickness), W/mK (thermal bridges)",
                },

                "climate_parameters": {
                    "coldest_month": 7,
                    "units": "1-12 (January-December)",
                },

                "heating_profile": {
                    "weekday": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,0.0],
                    "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,0.0],
                },
                "cooling_profile": {
                    "weekday": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
                    "weekend": [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,0.0],
                },
                "ventilation_profile": {
                    "weekday": [1.0] * 24,
                    "weekend": [1.0] * 24
                },
                "airflow_rates": {
                    "infiltration_rate": 1.0,   # ACH — old brick apartment, mid-floor, moderately leaky
                    "units": "ACH (air changes per hour)"
                }
            }
        }

@pytest.fixture
def output_dir():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_output = os.path.join(current_dir, "output_test")
    
    if not os.path.exists(test_output):
        os.makedirs(test_output)
        
    return test_output

def test_import_package():
    import pybuildingenergy as pybui
    assert hasattr(pybui, "__version__")

@pytest.mark.parametrize("fix", [True, False])
def test_sanitize_and_validate_bui(building_data, fix):
    import pybuildingenergy as pybui
    bui_result, report = pybui.sanitize_and_validate_BUI(building_data, fix=fix)
    assert bui_result is not None
    assert isinstance(report, list)
    errors = [e for e in report if e["level"] == "ERROR"]
    assert len(errors) == 0, f"Errors found: {errors}"

def test_iso52016_calculation(building_data, output_dir):
    import pybuildingenergy as pybui

    bui_checked, issues = pybui.sanitize_and_validate_BUI(building_data, fix=True)
    errors = [e for e in issues if e["level"] == "ERROR"]
    assert len(errors) == 0, "Errors in data validation"

    hourly_sim, annual_results_df, sankey_data = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
        bui_checked,
        weather_source="epw",
        path_weather_file=r"D:\ML+DL\pybuildingenergy\pyBuildingEnergy\tests\AUS_NSW.Sydney2025_IWEC.epw",
        # path_weather_file = None,
        occupants_schedule_workdays=bui_checked["building_parameters"]["internal_gains"][0]["weekday"],
        occupants_schedule_weekend=bui_checked["building_parameters"]["internal_gains"][0]["weekend"],
        appliances_schedule_workdays=bui_checked["building_parameters"]["internal_gains"][1]["weekday"],
        appliances_schedule_weekend=bui_checked["building_parameters"]["internal_gains"][1]["weekend"],
        lighting_schedule_workdays=bui_checked["building_parameters"]["internal_gains"][2]["weekday"],
        lighting_schedule_weekend=bui_checked["building_parameters"]["internal_gains"][2]["weekend"]
    )

    assert hourly_sim is not None
    assert annual_results_df is not None
    assert len(hourly_sim) > 0
    assert "x_air_in" in hourly_sim.columns, "Missing x_air_in column! Latent engine failed."
    assert "Q_Latent" in hourly_sim.columns, "Missing Q_Latent column! Latent engine failed."

    building_area = building_data["building"]["net_floor_area"]
    year = 2009

    country_calendar = generate_calendar("Victoria", year)
    n_working    = int((country_calendar["values"] == "Working").sum())
    n_nonworking = int((country_calendar["values"] == "Non-Working").sum())
    n_holiday    = int((country_calendar["values"] == "Holiday").sum())
    total_days   = len(country_calendar)

    hourly_fractions = pd.DataFrame({
        "Workday": [0.01, 0.01, 0.01, 0.01, 0.01, 0.02,
                    0.04, 0.06, 0.06, 0.04, 0.03, 0.04,
                    0.05, 0.04, 0.03, 0.03, 0.04, 0.06,
                    0.07, 0.07, 0.06, 0.05, 0.04, 0.02],
        "Weekend": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                    0.02, 0.04, 0.06, 0.07, 0.07, 0.06,
                    0.06, 0.05, 0.05, 0.04, 0.04, 0.05,
                    0.06, 0.06, 0.05, 0.04, 0.03, 0.02],
        "Holiday": [0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                    0.02, 0.04, 0.06, 0.07, 0.07, 0.06,
                    0.06, 0.05, 0.05, 0.04, 0.04, 0.05,
                    0.06, 0.06, 0.05, 0.04, 0.03, 0.02],
    })
    sum_fractions = pd.DataFrame(hourly_fractions.sum(), columns=["fractions"])

    (
        yearly_cons,
        V_W_nd_d,
        monthly_volume,
        yearly_volume,
        Q_W_nd_d,
        V_W_nd_h_i,
        daily_cons_volume,
        daily_cons_energy,
    ) = Volume_and_energy_DHW_calculation(
        n_workdays           = n_working,
        n_weekends           = n_nonworking,
        n_holidays           = n_holiday,
        sum_fractions        = sum_fractions,
        total_days           = total_days,
        hourly_fractions     = hourly_fractions,
        teta_W_draw          = 40.0,
        # teta_W_draw          = 45.0,
        teta_w_c_ref         = 10.0,
        teta_w_h_ref         = 60.0,
        teta_W_cold          = 10.0,
        mode_calc            = "volume_type_bui",
        building_type_B3     = None,
        building_area        = building_area,
        unit_count           = 1,
        building_type_B5     = "Dwelling",
        residential_typology = "apartments_dwellings - AVG",
        calculation_method   = "correlation",
        year                 = year,
        country_calendar     = country_calendar,
    )

    Q_DHW_annual_kWh = float(yearly_cons)
    Q_DHW_annual_Wh = Q_DHW_annual_kWh * 1000.0 

    Q_H_annual = float(annual_results_df["Q_H_annual"].iloc[0])
    Q_C_annual = float(annual_results_df["Q_C_annual"].iloc[0])
    Q_Latent_annual = float(hourly_sim["Q_Latent"].iloc[-8760:].sum())
    Q_total    = Q_H_annual + Q_C_annual + Q_DHW_annual_Wh + Q_Latent_annual

    assert Q_DHW_annual_kWh > 0, "DHW yearly energy should be positive"
    assert yearly_volume > 0,   "DHW yearly volume should be positive"
    assert Q_Latent_annual >= 0, "Latent annual should be non-negative"

    annual_results_df["Q_H_annual_kWh"] = Q_H_annual / 1000.0
    annual_results_df["Q_C_annual_kWh"] = Q_C_annual / 1000.0
    annual_results_df["Q_DHW_annual_kWh"] = Q_DHW_annual_kWh
    annual_results_df["Q_Latent_annual_kWh"] = Q_Latent_annual / 1000.0
    annual_results_df["Q_total_annual_kWh"]  = Q_total / 1000.0

    diff = len(hourly_sim) - len(daily_cons_energy)
    if diff > 0:
        dhw_energy_padded = daily_cons_energy[-diff:] + daily_cons_energy
        dhw_volume_padded = daily_cons_volume[-diff:] + daily_cons_volume
    else:
        dhw_energy_padded = daily_cons_energy
        dhw_volume_padded = daily_cons_volume

    hourly_sim["Q_DHW_Wh"] = dhw_energy_padded
    hourly_sim["V_DHW_m3"] = dhw_volume_padded

    hourly_sim_path = os.path.join(output_dir, "hourly_sim_test.csv")
    annual_sim_path = os.path.join(output_dir, "annual_results_test.csv")
    
    hourly_sim.to_csv(hourly_sim_path)
    annual_results_df.to_csv(annual_sim_path)

    assert os.path.exists(hourly_sim_path)
    assert os.path.exists(annual_sim_path)
    
    print("\nSETPOINTS USED IN SIMULATION:")
    print(bui_checked["building_parameters"]["temperature_setpoints"])