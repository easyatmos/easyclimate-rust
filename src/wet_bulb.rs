use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

const E_ZERO: f64 = 6.1078; // hPa, shared by the liquid-water and ice curves
const WATER: (f64, f64) = (17.2693882, 237.3);
const ICE: (f64, f64) = (22.46, 272.62); // WMO No. 8, Eq. (4.B.3)

#[derive(Debug, PartialEq)]
enum SolveError {
    InvalidInput,
    IceAboveFreezing,
    NotConverged,
}

impl SolveError {
    fn message(&self) -> &'static str {
        match self {
            Self::InvalidInput => "Invalid wet-bulb temperature input",
            Self::IceAboveFreezing => "No ice-bulb solution at or below 0 °C",
            Self::NotConverged => "Wet-bulb temperature did not converge",
        }
    }
}

fn saturation_pressure(t: f64, (b, c): (f64, f64)) -> f64 {
    E_ZERO * (b * (t / (t + c))).exp()
}

fn solve(
    t_dry: f64,
    rh: f64,
    p: f64,
    tolerance: f64,
    a: f64,
    max_iter: usize,
    ice: bool,
) -> Result<f64, SolveError> {
    let ap = a * p;
    if !t_dry.is_finite()
        || t_dry <= -WATER.1
        || !(0.0..=100.0).contains(&rh)
        || !p.is_finite()
        || p <= 0.0
        || !a.is_finite()
        || a <= 0.0
        || !ap.is_finite()
        || ap <= 0.0
        || !tolerance.is_finite()
        || tolerance <= 0.0
        || max_iter == 0
    {
        return Err(SolveError::InvalidInput);
    }

    // Ambient RH is relative to liquid water in both modes.
    let e_air = (rh / 100.0) * saturation_pressure(t_dry, WATER);
    let (b, c) = if ice { ICE } else { WATER };
    let mut t_wet = if ice { 0.0 } else { t_dry };
    // The residual slope is at least A * P, so this also bounds temperature error.
    let residual_tolerance = 1.0e-6_f64.min(tolerance * ap);

    for iteration in 0..=max_iter {
        let denominator = t_wet + c;
        if !denominator.is_finite() || denominator <= 0.0 {
            break;
        }
        let e_wet = saturation_pressure(t_wet, (b, c));
        let residual = e_wet - e_air + ap * (t_wet - t_dry);
        if !residual.is_finite() {
            break;
        }
        if residual.abs() <= residual_tolerance {
            return Ok(t_wet);
        }
        if ice && iteration == 0 && residual < 0.0 {
            return Err(SolveError::IceAboveFreezing);
        }
        if iteration == max_iter {
            break;
        }
        let slope = e_wet * b * (c / denominator) / denominator + ap;
        if !slope.is_finite() {
            break;
        }
        t_wet -= residual / slope;
    }

    Err(SolveError::NotConverged)
}

pub fn wet_bulb_temperature(
    t_dry: f64,
    rh: f64,
    p: f64,
    tolerance: f64,
    a: f64,
    max_iter: usize,
) -> Result<f64, &'static str> {
    solve(t_dry, rh, p, tolerance, a, max_iter, false).map_err(|error| error.message())
}

#[pyfunction]
pub fn calc_wet_bulb_temperature(
    t_dry: f64,
    rh: f64,
    p: f64,
    tolerance: f64,
    a: f64,
    max_iter: usize,
) -> PyResult<f64> {
    wet_bulb_temperature(t_dry, rh, p, tolerance, a, max_iter).map_err(PyRuntimeError::new_err)
}

#[pyfunction]
pub fn calc_wet_bulb_temperature_checked(
    t_dry: f64,
    rh: f64,
    p: f64,
    tolerance: f64,
    a: f64,
    max_iter: usize,
    ice: bool,
) -> PyResult<f64> {
    solve(t_dry, rh, p, tolerance, a, max_iter, ice).map_err(|error| match error {
        SolveError::InvalidInput | SolveError::IceAboveFreezing => {
            PyValueError::new_err(error.message())
        }
        SolveError::NotConverged => PyRuntimeError::new_err(error.message()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_saturation_pressure() {
        assert!((saturation_pressure(20.0, WATER) - 23.37).abs() < 0.1);
    }

    #[test]
    fn accepts_initial_and_final_estimates() {
        assert_eq!(
            wet_bulb_temperature(20.0, 100.0, 1013.25, 0.01, 0.000662, 1),
            Ok(20.0)
        );
        let wet = wet_bulb_temperature(20.0, 99.999, 1013.25, 0.01, 0.000662, 1)
            .expect("last update converges");
        assert!((wet - 19.99989).abs() < 1.0e-5);
        assert!(wet_bulb_temperature(20.0, 50.0, 1013.25, 0.01, 0.000662, 1).is_err());
    }

    #[test]
    fn ice_bulb_uses_ice_saturation_pressure() {
        let wet = solve(5.0, 20.0, 1000.0, 0.01, 0.000584, 100, true).unwrap();
        assert!((wet - (-1.36)).abs() < 0.01);
        assert_eq!(
            solve(20.0, 50.0, 1000.0, 0.01, 0.000584, 100, true),
            Err(SolveError::IceAboveFreezing)
        );
    }
}
