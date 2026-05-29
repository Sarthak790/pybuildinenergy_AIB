import numpy as np
import pandas as pd
import pytest
import os
from pathlib import Path


# ==============================================================================
#                           FIXTURES
# ==============================================================================

@pytest.fixture
def building_data():
    """Fixture for building data"""
    return {
        "building": {
            "name": "ML_Target_Building_001",
            "azimuth_relative_to_true_north": 41.8,
            "latitude": 46.49018685497359,
            "longitude": 11.327028776009655,
            "exposed_perimeter": 40,
            "height": 3,
            "wall_thickness": 0.3,
            "n_floors": 1,
            "building_type_class": "Residential_apartment",
            "adj_zones_present": False,
            "number_adj_zone": 2,
            "net_floor_area": 100,
            "construction_class": "class_i"
        },
        "adjacent_zones": [
            {
                "name": "adj_1",
                "orientation_zone": {"azimuth": 0},
                "area_facade_elements": np.array([20, 60, 30, 30, 50, 50], dtype=object),
                "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
                "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
                "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
                'volume': 300,
                'building_type_class': 'Residential_apartment',
                'a_use': 50
            },
            {
                "name": "adj_2",
                "orientation_zone": {"azimuth": 180},
                "area_facade_elements": np.array([20, 60, 30, 30, 50, 50], dtype=object),
                "typology_elements": np.array(['OP', 'OP', 'OP', 'OP', 'GR', 'OP'], dtype=object),
                "transmittance_U_elements": np.array([0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.8196721311475411, 0.5156683855612851, 1.162633192818565], dtype=object),
                "orientation_elements": np.array(['NV', 'SV', 'EV', 'WV', 'HOR', 'HOR'], dtype=object),
                'volume': 300,
                'building_type_class': 'Residential_apartment',
                'a_use': 50
            }
        ],
        "building_surface": [
            {
                "name": "Roof surface",
                "type": "opaque",
                "area": 130,
                "sky_view_factor": 1.0,
                "u_value": 2.2,
                "solar_absorptance": 0.4,
                "thermal_capacity": 741500.0,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None
            },
            {
                "name": "Opaque north surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.0,  # <-- Changed to 0.0 (meaning it sees no sky, it is buried)
                "basement_depth": 2.5,   # <-- NEW: The wall goes 2.5 meters deep
                "u_value": 1.4,
                "solar_absorptance": 0.4,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 0, "tilt": 90}, # Tilt 90 means it's vertical
                "name_adj_zone": "adj_1"
            },
            {
                "name": "Opaque south surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.4,
                "solar_absorptance": 0.4,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 180, "tilt": 90},
                "name_adj_zone": "adj_2"
            },
            {
                "name": "Opaque east surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.2,
                "solar_absorptance": 0.6,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 90, "tilt": 90},
                "name_adj_zone": None
            },
            {
                "name": "Opaque west surface",
                "type": "opaque",
                "area": 30,
                "sky_view_factor": 0.5,
                "u_value": 1.2,
                "solar_absorptance": 0.7,
                "thermal_capacity": 1416240.0,
                "orientation": {"azimuth": 270, "tilt": 90},
                "name_adj_zone": None
            },
            {
                "name": "Slab to ground",
                "type": "opaque",
                "area": 100,
                "sky_view_factor": 0.0,
                "u_value": 1.6,
                "solar_absorptance": 0.6,
                "thermal_capacity": 405801,
                "orientation": {"azimuth": 0, "tilt": 0},
                "name_adj_zone": None
            },
            {
                "name": "Transparent east surface",
                "type": "transparent",
                "area": 4,
                "sky_view_factor": 0.5,
                "u_value": 5,
                "g_value": 0.726,
                "height": 2,
                "width": 1,
                "parapet": 1.1,
                "orientation": {"azimuth": 90, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.5,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
                "name_adj_zone": None
            },
            {
                "name": "Transparent west surface",
                "type": "transparent",
                "area": 4,
                "sky_view_factor": 0.5,
                "u_value": 5,
                "g_value": 0.726,
                "height": 2,
                "width": 1,
                "parapet": 1.1,
                "orientation": {"azimuth": 270, "tilt": 90},
                "shading": False,
                "shading_type": "horizontal_overhang",
                "width_or_distance_of_shading_elements": 0.5,
                "overhang_proprieties": {"width_of_horizontal_overhangs": 1},
                "name_adj_zone": None
            }
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
                "heating_setpoint": 20.0,
                "heating_setback": 17.0,
                "cooling_setpoint": 26.0,
                "cooling_setback": 30.0,
                "units": "°C"
            },
            "system_capacities": {
                "heating_capacity": 10000000.0,
                "cooling_capacity": 12000000.0,
                "units": "W"
            },
            "airflow_rates": {
                "infiltration_rate": 1.0,
                "units": "ACH (air changes per hour)"
            },
            "internal_gains": [
                {
                    "name": "occupants",
                    "full_load": 4.2,
                    "weekday": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.8, 1.0, 1.0],
                    "weekend": [1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 1.0, 1.0]
                },
                {
                    "name": "appliances",
                    "full_load": 3,
                    "weekday": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
                    "weekend": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7, 0.7, 0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.5, 0.5, 0.7, 0.7, 0.8, 0.8, 0.8, 0.6, 0.6],
                },
                {
                    "name": "lighting",
                    "full_load": 3,
                    "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
                    "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.15, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.15, 0.15],
                }
            ],
            "construction": {
                "wall_thickness": 0.3,
                "thermal_bridges": 2,
                "units": "m (for thickness), W/mK (for thermal bridges)"
            },
            "climate_parameters": {
                "coldest_month": 1,
                "units": "1-12 (January-December)"
            },
            "heating_profile": {
                "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            },
            "cooling_profile": {
                "weekday": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                "weekend": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            },
            # === THE CORRECTED VENTILATION DICTIONARY ===
            "ventilation": {
                "ventilation_type": "custom", 
                # "type_ventilation": "custom", 
                "flow_rate_per_person": 0.005,
                "custom_heat_transfer_coefficient_ventilation": 25.0,
                "weekday": [1.0] * 24,
                "weekend": [1.0] * 24
            },
            "ventilation_profile": {
                "weekday": [1.0] * 24,
                "weekend": [1.0] * 24
            }
        }
    }


@pytest.fixture
def output_dir():
    """Fixture to save results directly to the Downloads folder"""
    # Use a raw string (r"...") for the Windows file path
    test_output = r"C:\Users\prakh\Downloads"
    
    # Optional but safe: check if it exists just in case
    if not os.path.exists(test_output):
        os.makedirs(test_output)
        
    return test_output
# ==============================================================================
#                           TESTS
# ==============================================================================

def test_import_package():
    """Test to verify package imports"""
    import pybuildingenergy as pybui
    assert hasattr(pybui, "__version__")


@pytest.mark.parametrize("fix", [True, False])
def test_sanitize_and_validate_bui(building_data, fix):
    """Test to validate building data schema"""
    import pybuildingenergy as pybui
    
    bui_result, report = pybui.sanitize_and_validate_BUI(building_data, fix=fix)
    
    assert bui_result is not None
    assert isinstance(report, list)
    
    # Verify no critical errors exist
    errors = [e for e in report if e["level"] == "ERROR"]
    assert len(errors) == 0, f"Errors found: {errors}"


# Removed the @pytest.mark.slow here so it will run by default when you run `pytest sample.py -v`
def test_iso52016_calculation(building_data, output_dir):
    """Test for the full ISO52016 calculation including Latent loads"""
    import pybuildingenergy as pybui
    
    # Data validation
    bui_checked, issues = pybui.sanitize_and_validate_BUI(building_data, fix=True)
    errors = [e for e in issues if e["level"] == "ERROR"]
    
    assert len(errors) == 0, "Errors in data validation"
    
    # Execute matrix calculation
    print("\nRunning the 8,760 hour thermal and latent simulation...")
    hourly_sim, annual_results_df, sankey_data = pybui.ISO52016.Temperature_and_Energy_needs_calculation(
        bui_checked,
        weather_source="pvgis"
    )
    
    # Core outputs verification
    assert hourly_sim is not None
    assert annual_results_df is not None
    assert len(hourly_sim) > 0
    assert len(annual_results_df) > 0
    
    # === NEW: Verify Latent Heat Engine is working ===
    assert "x_air_in" in hourly_sim.columns, "Missing x_air_in column! Latent engine failed."
    assert "Q_Latent" in hourly_sim.columns, "Missing Q_Latent column! Latent engine failed."
    
    print("\nLatent Engine Verification Passed!")
    print(f"Max Humidity Ratio tracked: {hourly_sim['x_air_in'].max():.4f}")
    print(f"Max Latent Cooling Load: {hourly_sim['Q_Latent'].max():.2f} W")
    
    # Save results
    hourly_sim_path = os.path.join(output_dir, "hourly_sim_test.csv")
    hourly_sim.to_csv(hourly_sim_path)
    annual_results_df.to_csv(os.path.join(output_dir, "annual_results_test.csv"))
    
    assert os.path.exists(hourly_sim_path)
    assert os.path.exists(os.path.join(output_dir, "annual_results_test.csv"))
    print(f"Simulation data successfully saved to: {hourly_sim_path}")
