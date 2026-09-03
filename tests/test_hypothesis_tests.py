import pytest
import numpy as np
import pandas as pd
from src.hypothesis_tests import run_adf_test, run_kpss_test, run_ljung_box_test, run_arch_lm_test

def test_adf_and_kpss_on_stationary_data():
    np.random.seed(42)
    stationary_data = pd.Series(np.random.normal(0, 1, 500))
    
    adf_res = run_adf_test(stationary_data)
    assert adf_res["is_stationary"] is True
    assert adf_res["p_value"] < 0.05
    
    kpss_res = run_kpss_test(stationary_data)
    assert kpss_res["is_stationary"] is True

def test_ljung_box_white_noise():
    np.random.seed(42)
    white_noise = pd.Series(np.random.normal(0, 1, 500))
    res = run_ljung_box_test(white_noise)
    assert res["is_white_noise"] is True
