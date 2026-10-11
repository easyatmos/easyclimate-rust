use nalgebra::DMatrix;
use num_complex::Complex64;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use rustfft::FftPlanner;
use std::collections::{HashMap, VecDeque};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

const DEFAULT_T: f64 = 50.0;
const DEFAULT_LY: f64 = 5.0 * PI;
const DEFAULT_LX: f64 = 12.0 * PI;
const DEFAULT_LZ: f64 = PI;
const DEFAULT_E: f64 = 0.1;
const DEFAULT_NT: usize = 500;
const DEFAULT_NY: usize = 60;
const DEFAULT_NX: usize = 128;
const DEFAULT_NZ: usize = 20;
const MAX_RAYON_THREADS: usize = 6;
const TRANSFORM_CACHE_CAPACITY: usize = 8;

type C64 = Complex64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CaseKind {
    GillSym,
    GillAsym,
    KelvinWave,
    GaussPert,
    MatsunoHeating,
    GillSymGaussianX,
    GillSymAsym,
    GillDeltaY1,
}

impl CaseKind {
    fn parse(name: &str) -> PyResult<Self> {
        let norm = name.trim().to_ascii_lowercase().replace('-', "_");
        match norm.as_str() {
            "1" | "gill_sym" | "sym" | "symmetric" | "symmetric_heating" => Ok(Self::GillSym),
            "2" | "gill_asym" | "asym" | "asymmetric" | "asymmetric_heating" => Ok(Self::GillAsym),
            "3" | "kelvin" | "kelvin_wave" => Ok(Self::KelvinWave),
            "4" | "gauss" | "gaussian" | "gauss_pert" | "gaussian_initial" => Ok(Self::GaussPert),
            "5" | "matsuno" | "matsuno_heating" | "forced_stationary" => Ok(Self::MatsunoHeating),
            "6" | "gill_sym_2" | "gill_sym_gaussian_x" | "symmetric_gaussian_x" => {
                Ok(Self::GillSymGaussianX)
            }
            "7" | "gill_sym_asym" | "sym_asym" | "symmetric_asymmetric" => Ok(Self::GillSymAsym),
            "8" | "gill_delta_y1" | "gill_sym_delta" | "delta_y1" => Ok(Self::GillDeltaY1),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown case '{name}'. Valid cases include: gill_sym, gill_asym, kelvin_wave, gauss_pert, matsuno_heating, gill_sym_gaussian_x, gill_sym_asym, gill_delta_y1"
            ))),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::GillSym => "Gill: symmetric heating",
            Self::GillAsym => "Gill: asymmetric heating",
            Self::KelvinWave => "Kelvin wave",
            Self::GaussPert => "Gaussian-shape initial perturbation",
            Self::MatsunoHeating => "Matsuno: forced stationary motion",
            Self::GillSymGaussianX => {
                "Gill: symmetric heating with Gaussian shape in zonal direction"
            }
            Self::GillSymAsym => "Gill: symmetric heating + asymmetric heating",
            Self::GillDeltaY1 => "Gill: delta heating at y = 1",
        }
    }

    fn short_name(self) -> &'static str {
        match self {
            Self::GillSym => "Gill_sym",
            Self::GillAsym => "Gill_asym",
            Self::KelvinWave => "KelvinWave",
            Self::GaussPert => "GaussPert",
            Self::MatsunoHeating => "MatsunoHeating",
            Self::GillSymGaussianX => "Gill_sym_2",
            Self::GillSymAsym => "Gill_sym_asym",
            Self::GillDeltaY1 => "Gill_sym_delta",
        }
    }
}

#[derive(Clone, Debug)]
struct Config {
    case: CaseKind,
    t: f64,
    lx: f64,
    ly: f64,
    lz: f64,
    damping: f64,
    nt: usize,
    nx: usize,
    ny: usize,
    nz: usize,
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct TransformCacheKey {
    nx: usize,
    ny: usize,
    dt_bits: u64,
    lx_bits: u64,
    ly_bits: u64,
    damping_bits: u64,
}

impl TransformCacheKey {
    fn from_config(cfg: &Config) -> Self {
        Self {
            nx: cfg.nx,
            ny: cfg.ny,
            dt_bits: (cfg.t / cfg.nt as f64).to_bits(),
            lx_bits: cfg.lx.to_bits(),
            ly_bits: cfg.ly.to_bits(),
            damping_bits: cfg.damping.to_bits(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kernel {
    Serial,
    Reyon,
}

impl Kernel {
    fn parse(name: &str) -> PyResult<Self> {
        match name.trim().to_ascii_lowercase().as_str() {
            "serial" => Ok(Self::Serial),
            "reyon" => Ok(Self::Reyon),
            _ => Err(pyo3::exceptions::PyValueError::new_err(
                "unknown kernel. Valid kernels are: serial, reyon",
            )),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::Reyon => "reyon",
        }
    }
}

#[derive(Clone)]
struct TransformSet {
    aeta_v: DMatrix<C64>,
    au_v: DMatrix<C64>,
    av_v: DMatrix<C64>,
    aqv: DMatrix<C64>,
    avv_eta: DMatrix<C64>,
    aeta_eta: C64,
    au_eta: C64,
    aqeta: C64,
    avv_u: DMatrix<C64>,
    aeta_u: C64,
    au_u: C64,
    aqu: C64,
}

struct SpectralFields {
    v: Vec<C64>,
    u: Vec<C64>,
    eta: Vec<C64>,
}

struct SpectralMode<'a> {
    v: &'a [C64],
    u: &'a [C64],
    eta: &'a [C64],
}

struct SpectralModeMut<'a> {
    v: &'a mut [C64],
    u: &'a mut [C64],
    eta: &'a mut [C64],
}

impl SpectralFields {
    fn zeros(nx: usize, ny: usize) -> Self {
        Self {
            v: vec![C64::new(0.0, 0.0); nx * ny],
            u: vec![C64::new(0.0, 0.0); nx * (ny + 1)],
            eta: vec![C64::new(0.0, 0.0); nx * (ny + 1)],
        }
    }
}

fn idx(row: usize, col: usize, ncols: usize) -> usize {
    row * ncols + col
}

fn eye(n: usize) -> DMatrix<C64> {
    DMatrix::<C64>::identity(n, n)
}

fn transpose(m: &DMatrix<C64>) -> DMatrix<C64> {
    let mut out = DMatrix::<C64>::zeros(m.ncols(), m.nrows());
    for r in 0..m.nrows() {
        for c in 0..m.ncols() {
            out[(c, r)] = m[(r, c)];
        }
    }
    out
}

fn make_base_matrices(
    ny: usize,
    ly: f64,
) -> (DMatrix<C64>, DMatrix<C64>, DMatrix<C64>, DMatrix<C64>) {
    let nyp = ny + 1;
    let dy = ly / (ny as f64 + 1.0);
    let mut a13 = DMatrix::<C64>::zeros(nyp, ny);
    let mut a23 = DMatrix::<C64>::zeros(nyp, ny);

    for j in 0..ny {
        // A13 = ([I;0] + [0;-I])/(2*dy)
        a13[(j, j)] += C64::new(1.0 / (2.0 * dy), 0.0);
        a13[(j + 1, j)] += C64::new(-1.0 / (2.0 * dy), 0.0);

        // A23 = -([diag(-Ny/2:Ny/2-1);0] + [0;diag(-Ny/2+1:Ny/2)])*dy/8
        let c1 = -((j as f64) - ny as f64 / 2.0) * dy / 8.0;
        let c2 = -((j as f64) - ny as f64 / 2.0 + 1.0) * dy / 8.0;
        a23[(j, j)] += C64::new(c1, 0.0);
        a23[(j + 1, j)] += C64::new(c2, 0.0);
    }

    let a31 = transpose(&a13) * C64::new(-1.0, 0.0);
    let a32 = transpose(&a23) * C64::new(-1.0, 0.0);
    (a13, a31, a23, a32)
}

fn matgen(cfg: &Config) -> PyResult<Vec<TransformSet>> {
    let ny = cfg.ny;
    let nx = cfg.nx;
    let dt = cfg.t / cfg.nt as f64;
    let alpha = 1.0 / dt + cfg.damping / 2.0;
    let beta = 1.0 / dt - cfg.damping / 2.0;
    let half = C64::new(0.5, 0.0);

    let (a13, a31, a23, a32) = make_base_matrices(ny, cfg.ly);
    let b_plus = &a31 + &a32;
    let b_minus = &a31 - &a32;
    let c_plus = &a13 + &a23;
    let c_minus = &a13 - &a23;
    let eye_ny = eye(ny);

    let mut out = Vec::with_capacity(nx);
    for i in 0..nx {
        let wave_index = if i <= nx / 2 {
            i as f64
        } else {
            i as f64 - nx as f64
        };
        let k = 2.0 * PI / cfg.lx * wave_index;
        let ik2 = C64::new(0.0, k / 2.0);
        let e_plus_k = C64::new(alpha, 0.0) + ik2;
        let e_minus_k = C64::new(alpha, 0.0) - ik2;
        let ebar_minus_k = C64::new(beta, 0.0) - ik2;
        let ebar_plus_k = C64::new(beta, 0.0) + ik2;
        let inv_plus = C64::new(1.0, 0.0) / e_plus_k;
        let inv_minus = C64::new(1.0, 0.0) / e_minus_k;

        // Av_inv = inv(M.'), where .' is a non-conjugate transpose.
        let m = eye_ny.clone() * C64::new(alpha, 0.0)
            - (&b_plus * &c_plus) * (inv_plus * half)
            - (&b_minus * &c_minus) * (inv_minus * half);
        let av_inv = transpose(&m).try_inverse().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "singular v-transform matrix at zonal index {i}"
            ))
        })?;

        let expr_eta_v = a31.clone() * C64::new(-1.0, 0.0)
            - b_plus.clone() * (inv_plus * ebar_minus_k * half)
            - b_minus.clone() * (inv_minus * ebar_plus_k * half);
        let expr_u_v = a32.clone() * C64::new(-1.0, 0.0)
            - b_plus.clone() * (inv_plus * ebar_minus_k * half)
            + b_minus.clone() * (inv_minus * ebar_plus_k * half);
        let expr_v_v = eye_ny.clone() * C64::new(beta, 0.0)
            + (&b_plus * &c_plus) * (inv_plus * half)
            + (&b_minus * &c_minus) * (inv_minus * half);
        let expr_q_v = b_plus.clone() * (inv_plus * half) + b_minus.clone() * (inv_minus * half);

        let scalar_eta = (inv_plus * ebar_minus_k + inv_minus * ebar_plus_k) * half;
        let scalar_u = (inv_plus * ebar_minus_k - inv_minus * ebar_plus_k) * half;
        let scalar_q_eta = -(inv_plus + inv_minus) * half;
        let scalar_q_u = -(inv_plus - inv_minus) * half;

        out.push(TransformSet {
            aeta_v: transpose(&expr_eta_v) * &av_inv,
            au_v: transpose(&expr_u_v) * &av_inv,
            av_v: transpose(&expr_v_v) * &av_inv,
            aqv: transpose(&expr_q_v) * &av_inv,
            avv_eta: transpose(&(c_plus.clone() * inv_plus + c_minus.clone() * inv_minus))
                * C64::new(-1.0, 0.0),
            aeta_eta: scalar_eta,
            au_eta: scalar_u,
            aqeta: scalar_q_eta,
            avv_u: transpose(&(c_plus.clone() * inv_plus - c_minus.clone() * inv_minus))
                * C64::new(-1.0, 0.0),
            aeta_u: scalar_u,
            au_u: scalar_eta,
            aqu: scalar_q_u,
        });
    }

    Ok(out)
}

fn transforms_for(cfg: &Config) -> PyResult<Arc<Vec<TransformSet>>> {
    let key = TransformCacheKey::from_config(cfg);
    let cache = TRANSFORM_CACHE.get_or_init(|| Mutex::new(TransformCache::default()));

    {
        let cache = cache.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("Gill transform cache lock was poisoned")
        })?;
        if let Some(transforms) = cache.entries.get(&key) {
            return Ok(Arc::clone(transforms));
        }
    }

    let transforms = Arc::new(matgen(cfg)?);
    let mut cache = cache.lock().map_err(|_| {
        pyo3::exceptions::PyRuntimeError::new_err("Gill transform cache lock was poisoned")
    })?;

    if let Some(existing) = cache.entries.get(&key) {
        return Ok(Arc::clone(existing));
    }
    if cache.entries.len() >= TRANSFORM_CACHE_CAPACITY {
        if let Some(evicted_key) = cache.insertion_order.pop_front() {
            cache.entries.remove(&evicted_key);
        }
    }
    cache.insertion_order.push_back(key.clone());
    cache.entries.insert(key, Arc::clone(&transforms));
    Ok(transforms)
}

fn row_mat_mul_into(row: &[C64], mat: &DMatrix<C64>, out: &mut [C64]) {
    debug_assert_eq!(row.len(), mat.nrows());
    debug_assert_eq!(out.len(), mat.ncols());
    for c in 0..mat.ncols() {
        let mut acc = C64::new(0.0, 0.0);
        for r in 0..mat.nrows() {
            acc += row[r] * mat[(r, c)];
        }
        out[c] = acc;
    }
}

fn row_mat_mul_add(row: &[C64], mat: &DMatrix<C64>, out: &mut [C64]) {
    debug_assert_eq!(row.len(), mat.nrows());
    debug_assert_eq!(out.len(), mat.ncols());
    for c in 0..mat.ncols() {
        let mut acc = C64::new(0.0, 0.0);
        for r in 0..mat.nrows() {
            acc += row[r] * mat[(r, c)];
        }
        out[c] += acc;
    }
}

fn averaged_row_mat_mul_into(
    new_row: &[C64],
    old_row: &[C64],
    mat: &DMatrix<C64>,
    out: &mut [C64],
) {
    debug_assert_eq!(new_row.len(), old_row.len());
    debug_assert_eq!(new_row.len(), mat.nrows());
    debug_assert_eq!(out.len(), mat.ncols());
    let half = C64::new(0.5, 0.0);
    for c in 0..mat.ncols() {
        let mut acc = C64::new(0.0, 0.0);
        for r in 0..mat.nrows() {
            acc += ((new_row[r] + old_row[r]) * half) * mat[(r, c)];
        }
        out[c] = acc;
    }
}

fn add_scaled(dst: &mut [C64], src: &[C64], scale: C64) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src) {
        *d += *s * scale;
    }
}

fn add_assign(dst: &mut [C64], src: &[C64]) {
    debug_assert_eq!(dst.len(), src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d += *s;
    }
}

fn fft_along_x(real_data: &[f64], nx: usize, ncols: usize) -> Vec<C64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(nx);
    let mut out = vec![C64::new(0.0, 0.0); nx * ncols];
    let mut buffer = vec![C64::new(0.0, 0.0); nx];

    for col in 0..ncols {
        for i in 0..nx {
            buffer[i] = C64::new(real_data[idx(i, col, ncols)], 0.0);
        }
        fft.process(&mut buffer);
        for i in 0..nx {
            out[idx(i, col, ncols)] = buffer[i];
        }
    }
    out
}

fn ifft_along_x(complex_data: &[C64], nx: usize, ncols: usize) -> Vec<f64> {
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_inverse(nx);
    let mut out = vec![0.0; nx * ncols];
    let mut buffer = vec![C64::new(0.0, 0.0); nx];
    let scale = nx as f64;

    for col in 0..ncols {
        for i in 0..nx {
            buffer[i] = complex_data[idx(i, col, ncols)];
        }
        fft.process(&mut buffer);
        for i in 0..nx {
            out[idx(i, col, ncols)] = buffer[i].re / scale;
        }
    }
    out
}

fn nested2(data: &[f64], nrows: usize, ncols: usize) -> Vec<Vec<f64>> {
    let mut out = Vec::with_capacity(nrows);
    for r in 0..nrows {
        out.push(data[r * ncols..(r + 1) * ncols].to_vec());
    }
    out
}

fn make_grid(cfg: &Config) -> (Vec<f64>, Vec<f64>, Vec<f64>, f64, f64, f64) {
    let dy = cfg.ly / (cfg.ny as f64 + 1.0);
    let dx = cfg.lx / cfg.nx as f64;
    let dz = cfg.lz / (cfg.nz as f64 - 1.0);

    let mut x: Vec<f64> = (0..cfg.nx)
        .map(|i| dx * (i as f64 - cfg.nx as f64 / 2.0))
        .collect();
    if matches!(
        cfg.case,
        CaseKind::GillSym
            | CaseKind::GillSymGaussianX
            | CaseKind::GillSymAsym
            | CaseKind::GillDeltaY1
    ) {
        for xx in x.iter_mut() {
            *xx += 2.0 * PI;
        }
    }

    let y: Vec<f64> = (0..=cfg.ny)
        .map(|j| dy * (j as f64 - cfg.ny as f64 / 2.0))
        .collect();
    let z: Vec<f64> = (0..cfg.nz).map(|k| dz * k as f64).collect();
    (x, y, z, dx, dy, dz)
}

fn initial_conditions(
    cfg: &Config,
    x: &[f64],
    y: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let nx = cfg.nx;
    let ny = cfg.ny;
    let nyp = ny + 1;
    let mut v0 = vec![0.0; nx * ny];
    let mut u0 = vec![0.0; nx * nyp];
    let mut eta0 = vec![0.0; nx * nyp];
    let mut q = vec![0.0; nx * nyp];

    match cfg.case {
        CaseKind::GillSym => {
            let l = 2.0;
            for i in 0..nx {
                let fx = if x[i].abs() < l {
                    (PI / (2.0 * l) * x[i]).cos()
                } else {
                    0.0
                };
                for j in 0..nyp {
                    q[idx(i, j, nyp)] = fx * (-(y[j] * y[j]) / 4.0).exp();
                }
            }
        }
        CaseKind::GillAsym => {
            let l = 2.0;
            for i in 0..nx {
                let fx = if x[i].abs() < l {
                    (PI / (2.0 * l) * x[i]).cos()
                } else {
                    0.0
                };
                for j in 0..nyp {
                    q[idx(i, j, nyp)] = fx * y[j] * (-(y[j] * y[j]) / 4.0).exp();
                }
            }
        }
        CaseKind::KelvinWave => {
            for i in 0..nx {
                let xpart = (4.0 * 2.0 * PI / cfg.lx * x[i]).cos();
                for j in 0..nyp {
                    let val = xpart * (-(y[j] * y[j]) / 4.0).exp();
                    eta0[idx(i, j, nyp)] = val;
                    u0[idx(i, j, nyp)] = val;
                }
            }
        }
        CaseKind::GaussPert => {
            for i in 0..nx {
                let xpart = (-(x[i] * x[i]) / 4.0).exp();
                for j in 0..nyp {
                    eta0[idx(i, j, nyp)] = xpart * (-(y[j] * y[j]) / 4.0).exp();
                }
            }
        }
        CaseKind::MatsunoHeating => {
            for i in 0..nx {
                let xpart = (0.5 * x[i]).sin();
                for j in 0..nyp {
                    q[idx(i, j, nyp)] = xpart * (-(y[j] * y[j]) / 4.0).exp();
                }
            }
        }
        CaseKind::GillSymGaussianX => {
            for i in 0..nx {
                let xpart = (-(x[i] * x[i]) / 2.0).exp();
                for j in 0..nyp {
                    q[idx(i, j, nyp)] = xpart * (-(y[j] * y[j]) / 4.0).exp();
                }
            }
        }
        CaseKind::GillSymAsym => {
            let l = 2.0;
            for i in 0..nx {
                let fx = if x[i].abs() < l {
                    (PI / (2.0 * l) * x[i]).cos()
                } else {
                    0.0
                };
                for j in 0..nyp {
                    let gauss_y = (-(y[j] * y[j]) / 4.0).exp();
                    q[idx(i, j, nyp)] = fx * (gauss_y + y[j] * gauss_y);
                }
            }
        }
        CaseKind::GillDeltaY1 => {
            let l = 2.0;
            for i in 0..nx {
                let fx = if x[i].abs() < l {
                    (PI / (2.0 * l) * x[i]).cos()
                } else {
                    0.0
                };
                for j in 0..nyp {
                    let fy = if (y[j] - 1.0).abs() < 0.5 { 1.0 } else { 0.0 };
                    q[idx(i, j, nyp)] = fy + fx * (-(y[j] * y[j]) / 4.0).exp();
                }
            }
        }
    }

    // v0 is currently zero for all MATLAB examples, but it is kept here to preserve the staggered-grid layout.
    let _ = &mut v0;
    (v0, u0, eta0, q)
}

fn v_to_eta_grid(v_raw: &[f64], nx: usize, ny: usize) -> Vec<f64> {
    let nyp = ny + 1;
    let mut out = vec![0.0; nx * nyp];
    for i in 0..nx {
        for j in 0..nyp {
            let left = if j < ny { v_raw[idx(i, j, ny)] } else { 0.0 };
            let right = if j > 0 { v_raw[idx(i, j - 1, ny)] } else { 0.0 };
            out[idx(i, j, nyp)] = 0.5 * (left + right);
        }
    }
    out
}

fn build_config(
    case_name: &str,
    t: Option<f64>,
    nt: Option<usize>,
    nx: usize,
    ny: usize,
    nz: usize,
    lx: Option<f64>,
    ly: Option<f64>,
    lz: Option<f64>,
    damping: Option<f64>,
) -> PyResult<Config> {
    let case = CaseKind::parse(case_name)?;
    let mut cfg = Config {
        case,
        t: t.unwrap_or(DEFAULT_T),
        lx: lx.unwrap_or(DEFAULT_LX),
        ly: ly.unwrap_or(DEFAULT_LY),
        lz: lz.unwrap_or(DEFAULT_LZ),
        damping: damping.unwrap_or(DEFAULT_E),
        nt: nt.unwrap_or(DEFAULT_NT),
        nx,
        ny,
        nz,
    };

    match case {
        CaseKind::KelvinWave => {
            cfg.t = t.unwrap_or(10.0);
            cfg.nt = nt.unwrap_or(100);
            cfg.damping = damping.unwrap_or(0.0);
        }
        CaseKind::GaussPert => {
            cfg.damping = damping.unwrap_or(0.0);
        }
        CaseKind::MatsunoHeating => {
            cfg.damping = damping.unwrap_or(0.2);
        }
        _ => {}
    }

    if cfg.nt == 0 || cfg.nx == 0 || cfg.ny == 0 || cfg.nz < 2 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "nt, nx, ny must be positive and nz must be at least 2",
        ));
    }
    Ok(cfg)
}

#[derive(Default)]
struct TransformCache {
    entries: HashMap<TransformCacheKey, Arc<Vec<TransformSet>>>,
    insertion_order: VecDeque<TransformCacheKey>,
}

static TRANSFORM_CACHE: OnceLock<Mutex<TransformCache>> = OnceLock::new();
static REYON_THREAD_POOL: OnceLock<Result<rayon::ThreadPool, String>> = OnceLock::new();

fn precompute_forcing(qk: &[C64], transforms: &[TransformSet], cfg: &Config) -> SpectralFields {
    let nyp = cfg.ny + 1;
    assert_eq!(
        transforms.len(),
        cfg.nx,
        "one transform is required per zonal mode"
    );
    assert_eq!(
        qk.len(),
        cfg.nx * nyp,
        "forcing spectrum has an unexpected shape"
    );
    let mut forcing = SpectralFields::zeros(cfg.nx, cfg.ny);

    for (i, tr) in transforms.iter().enumerate() {
        let q_row = &qk[i * nyp..(i + 1) * nyp];
        row_mat_mul_into(q_row, &tr.aqv, &mut forcing.v[i * cfg.ny..(i + 1) * cfg.ny]);
        for (j, q) in q_row.iter().enumerate() {
            forcing.u[i * nyp + j] = *q * tr.aqu;
            forcing.eta[i * nyp + j] = *q * tr.aqeta;
        }
    }

    forcing
}

fn advance_mode(
    current: SpectralMode<'_>,
    next: SpectralModeMut<'_>,
    forcing: SpectralMode<'_>,
    tr: &TransformSet,
) {
    row_mat_mul_into(current.eta, &tr.aeta_v, next.v);
    row_mat_mul_add(current.u, &tr.au_v, next.v);
    row_mat_mul_add(current.v, &tr.av_v, next.v);
    add_assign(next.v, forcing.v);

    averaged_row_mat_mul_into(next.v, current.v, &tr.avv_u, next.u);
    add_scaled(next.u, current.eta, tr.aeta_u);
    add_scaled(next.u, current.u, tr.au_u);
    add_assign(next.u, forcing.u);

    averaged_row_mat_mul_into(next.v, current.v, &tr.avv_eta, next.eta);
    add_scaled(next.eta, current.eta, tr.aeta_eta);
    add_scaled(next.eta, next.u, tr.au_eta);
    add_assign(next.eta, forcing.eta);
}

fn advance_step_serial(
    current: &SpectralFields,
    next: &mut SpectralFields,
    forcing: &SpectralFields,
    transforms: &[TransformSet],
    cfg: &Config,
) {
    let nyp = cfg.ny + 1;

    for (i, tr) in transforms.iter().enumerate() {
        let v_range = i * cfg.ny..(i + 1) * cfg.ny;
        let field_range = i * nyp..(i + 1) * nyp;
        advance_mode(
            SpectralMode {
                v: &current.v[v_range.clone()],
                u: &current.u[field_range.clone()],
                eta: &current.eta[field_range.clone()],
            },
            SpectralModeMut {
                v: &mut next.v[v_range.clone()],
                u: &mut next.u[field_range.clone()],
                eta: &mut next.eta[field_range.clone()],
            },
            SpectralMode {
                v: &forcing.v[v_range],
                u: &forcing.u[field_range.clone()],
                eta: &forcing.eta[field_range],
            },
            tr,
        );
    }
}

fn advance_step_reyon(
    current: &SpectralFields,
    next: &mut SpectralFields,
    forcing: &SpectralFields,
    transforms: &[TransformSet],
    cfg: &Config,
) {
    let nyp = cfg.ny + 1;
    next.v
        .par_chunks_mut(cfg.ny)
        .zip(next.u.par_chunks_mut(nyp))
        .zip(next.eta.par_chunks_mut(nyp))
        .enumerate()
        .for_each(|(i, ((v_new, u_new), eta_new))| {
            let v_range = i * cfg.ny..(i + 1) * cfg.ny;
            let field_range = i * nyp..(i + 1) * nyp;
            advance_mode(
                SpectralMode {
                    v: &current.v[v_range.clone()],
                    u: &current.u[field_range.clone()],
                    eta: &current.eta[field_range.clone()],
                },
                SpectralModeMut {
                    v: v_new,
                    u: u_new,
                    eta: eta_new,
                },
                SpectralMode {
                    v: &forcing.v[v_range],
                    u: &forcing.u[field_range.clone()],
                    eta: &forcing.eta[field_range],
                },
                &transforms[i],
            );
        });
}

fn reyon_thread_pool() -> PyResult<&'static rayon::ThreadPool> {
    REYON_THREAD_POOL
        .get_or_init(|| {
            ThreadPoolBuilder::new()
                .num_threads(MAX_RAYON_THREADS)
                .build()
                .map_err(|err| err.to_string())
        })
        .as_ref()
        .map_err(|err| pyo3::exceptions::PyRuntimeError::new_err(err.clone()))
}

#[pyfunction]
#[pyo3(signature = (
    case = "gill_sym",
    t = None,
    nt = None,
    nx = 128,
    ny = 60,
    nz = 20,
    lx = None,
    ly = None,
    lz = None,
    damping = None,
    store_every = 0,
    kernel = "serial"
))]
pub fn gill_simulate(
    py: Python<'_>,
    case: &str,
    t: Option<f64>,
    nt: Option<usize>,
    nx: usize,
    ny: usize,
    nz: usize,
    lx: Option<f64>,
    ly: Option<f64>,
    lz: Option<f64>,
    damping: Option<f64>,
    store_every: usize,
    kernel: &str,
) -> PyResult<Py<PyDict>> {
    let cfg = build_config(case, t, nt, nx, ny, nz, lx, ly, lz, damping)?;
    let kernel = Kernel::parse(kernel)?;
    let nyp = cfg.ny + 1;
    let dt = cfg.t / cfg.nt as f64;

    let (x, y, z, dx, dy, dz) = make_grid(&cfg);
    let (v0, u0, eta0, q) = initial_conditions(&cfg, &x, &y);
    let transforms = transforms_for(&cfg)?;

    let mut current = SpectralFields {
        v: fft_along_x(&v0, cfg.nx, cfg.ny),
        u: fft_along_x(&u0, cfg.nx, nyp),
        eta: fft_along_x(&eta0, cfg.nx, nyp),
    };
    let mut next = SpectralFields::zeros(cfg.nx, cfg.ny);
    let qk = fft_along_x(&q, cfg.nx, nyp);
    let forcing = precompute_forcing(&qk, transforms.as_ref(), &cfg);

    let mut saved_time: Vec<f64> = Vec::new();
    let mut eta_history: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut u_history: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut v_history: Vec<Vec<Vec<f64>>> = Vec::new();

    let rayon_pool = if kernel == Kernel::Reyon {
        Some(reyon_thread_pool()?)
    } else {
        None
    };

    for step in 1..=cfg.nt {
        match &rayon_pool {
            Some(pool) => pool.install(|| {
                advance_step_reyon(&current, &mut next, &forcing, transforms.as_ref(), &cfg)
            }),
            None => advance_step_serial(&current, &mut next, &forcing, transforms.as_ref(), &cfg),
        }
        std::mem::swap(&mut current, &mut next);

        if store_every > 0 && (step % store_every == 0 || step == cfg.nt) {
            let eta = ifft_along_x(&current.eta, cfg.nx, nyp);
            let u = ifft_along_x(&current.u, cfg.nx, nyp);
            let v_raw = ifft_along_x(&current.v, cfg.nx, cfg.ny);
            let v = v_to_eta_grid(&v_raw, cfg.nx, cfg.ny);
            saved_time.push(step as f64 * dt);
            eta_history.push(nested2(&eta, cfg.nx, nyp));
            u_history.push(nested2(&u, cfg.nx, nyp));
            v_history.push(nested2(&v, cfg.nx, nyp));
        }
    }

    let eta = ifft_along_x(&current.eta, cfg.nx, nyp);
    let u = ifft_along_x(&current.u, cfg.nx, nyp);
    let v_staggered = ifft_along_x(&current.v, cfg.nx, cfg.ny);
    let v = v_to_eta_grid(&v_staggered, cfg.nx, cfg.ny);

    let dict = PyDict::new(py);
    dict.set_item("case", case)?;
    dict.set_item("case_label", cfg.case.label())?;
    dict.set_item("short_name", cfg.case.short_name())?;
    dict.set_item("t", cfg.t)?;
    dict.set_item("nt", cfg.nt)?;
    dict.set_item("nx", cfg.nx)?;
    dict.set_item("ny", cfg.ny)?;
    dict.set_item("nz", cfg.nz)?;
    dict.set_item("lx", cfg.lx)?;
    dict.set_item("ly", cfg.ly)?;
    dict.set_item("lz", cfg.lz)?;
    dict.set_item("damping", cfg.damping)?;
    dict.set_item("kernel", kernel.name())?;
    dict.set_item("dt", dt)?;
    dict.set_item("dx", dx)?;
    dict.set_item("dy", dy)?;
    dict.set_item("dz", dz)?;
    dict.set_item("x", x)?;
    dict.set_item("y", y)?;
    dict.set_item("z", z)?;
    dict.set_item("q", nested2(&q, cfg.nx, nyp))?;
    dict.set_item("eta", nested2(&eta, cfg.nx, nyp))?;
    dict.set_item("u", nested2(&u, cfg.nx, nyp))?;
    dict.set_item("v", nested2(&v, cfg.nx, nyp))?;
    dict.set_item("v_staggered", nested2(&v_staggered, cfg.nx, cfg.ny))?;
    dict.set_item("time", saved_time)?;
    dict.set_item("eta_history", eta_history)?;
    dict.set_item("u_history", u_history)?;
    dict.set_item("v_history", v_history)?;
    Ok(dict.unbind())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use pyo3::exceptions::PyValueError;
    use std::f64::consts::PI;

    fn c(re: f64) -> C64 {
        C64::new(re, 0.0)
    }

    fn make_cfg(case: CaseKind, nx: usize, ny: usize) -> Config {
        Config {
            case,
            t: 50.0,
            lx: 12.0 * PI,
            ly: 5.0 * PI,
            lz: PI,
            damping: 0.1,
            nt: 500,
            nx,
            ny,
            nz: 4,
        }
    }

    fn assert_all_zero(v: &[f64]) {
        assert!(v.iter().all(|&x| x == 0.0));
    }

    fn assert_nonzero(v: &[f64]) {
        assert!(v.iter().any(|&x| x != 0.0));
    }

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "values differ: {a} vs {b} (tol {tol})");
    }

    fn weighted_2d(values: &[Vec<f64>]) -> f64 {
        values
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(j, value)| value * ((i + 1) * 17 + (j + 1) * 31) as f64)
            })
            .sum()
    }

    fn weighted_3d(values: &[Vec<Vec<f64>>]) -> f64 {
        values
            .iter()
            .enumerate()
            .flat_map(|(t, plane)| {
                plane.iter().enumerate().flat_map(move |(i, row)| {
                    row.iter().enumerate().map(move |(j, value)| {
                        value * ((t + 1) * 13 + (i + 1) * 17 + (j + 1) * 31) as f64
                    })
                })
            })
            .sum()
    }

    // ---------- idx / eye / transpose ----------

    #[test]
    fn test_idx() {
        assert_eq!(idx(2, 3, 5), 13);
        assert_eq!(idx(0, 0, 1), 0);
    }

    #[test]
    fn test_eye() {
        let m = eye(3);
        assert_eq!(m.nrows(), 3);
        assert_eq!(m.ncols(), 3);
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(m[(i, j)], if i == j { c(1.0) } else { c(0.0) });
            }
        }
    }

    #[test]
    fn test_transpose() {
        let mut m = DMatrix::<C64>::zeros(2, 3);
        m[(0, 1)] = c(2.0);
        m[(1, 2)] = c(5.0);
        let t = transpose(&m);
        assert_eq!(t.nrows(), 3);
        assert_eq!(t.ncols(), 2);
        assert_eq!(t[(1, 0)], c(2.0));
        assert_eq!(t[(2, 1)], c(5.0));
        assert_eq!(t[(0, 0)], c(0.0));
    }

    // ---------- make_base_matrices ----------

    #[test]
    fn test_make_base_matrices_values() {
        let ny = 4;
        let ly = 10.0;
        let (a13, a31, a23, a32) = make_base_matrices(ny, ly);
        let dy = ly / (ny as f64 + 1.0);

        assert_eq!(a13.nrows(), 5);
        assert_eq!(a13.ncols(), 4);
        assert_eq!(a23.nrows(), 5);
        assert_eq!(a23.ncols(), 4);
        assert_eq!(a31.nrows(), 4);
        assert_eq!(a31.ncols(), 5);
        assert_eq!(a32.nrows(), 4);
        assert_eq!(a32.ncols(), 5);

        assert_eq!(a13[(0, 0)], c(1.0 / (2.0 * dy)));
        assert_eq!(a13[(1, 0)], c(-1.0 / (2.0 * dy)));
        let c1 = -((0.0_f64) - ny as f64 / 2.0) * dy / 8.0;
        let c2 = -((0.0_f64) - ny as f64 / 2.0 + 1.0) * dy / 8.0;
        assert_eq!(a23[(0, 0)], c(c1));
        assert_eq!(a23[(1, 0)], c(c2));
        assert_eq!(a31[(0, 0)], -a13[(0, 0)]);
        assert_eq!(a31[(0, 1)], -a13[(1, 0)]);
        assert_eq!(a32[(0, 0)], -a23[(0, 0)]);
        assert_eq!(a32[(0, 1)], -a23[(1, 0)]);
    }

    // ---------- matgen ----------

    #[test]
    fn test_matgen_shapes_and_count() {
        let cfg = Config {
            case: CaseKind::GillSym,
            t: 10.0,
            lx: 12.0 * PI,
            ly: 5.0 * PI,
            lz: PI,
            damping: 0.1,
            nt: 10,
            nx: 8,
            ny: 6,
            nz: 4,
        };
        let transforms = matgen(&cfg).unwrap();
        assert_eq!(transforms.len(), 8);
        for tr in &transforms {
            assert_eq!(tr.aeta_v.nrows(), 7);
            assert_eq!(tr.aeta_v.ncols(), 6);
            assert_eq!(tr.au_v.nrows(), 7);
            assert_eq!(tr.au_v.ncols(), 6);
            assert_eq!(tr.av_v.nrows(), 6);
            assert_eq!(tr.av_v.ncols(), 6);
            assert_eq!(tr.aqv.nrows(), 7);
            assert_eq!(tr.aqv.ncols(), 6);
            assert_eq!(tr.avv_eta.nrows(), 6);
            assert_eq!(tr.avv_eta.ncols(), 7);
            assert_eq!(tr.avv_u.nrows(), 6);
            assert_eq!(tr.avv_u.ncols(), 7);
            for scalar in [tr.aeta_eta, tr.au_eta, tr.aqeta, tr.aeta_u, tr.au_u, tr.aqu] {
                assert!(scalar.re.is_finite());
                assert!(scalar.im.is_finite());
            }
        }
    }

    // ---------- row matrix operations ----------

    #[test]
    fn test_row_mat_mul() {
        let row = vec![c(1.0), c(2.0)];
        let mut mat = DMatrix::<C64>::zeros(2, 3);
        mat[(0, 0)] = c(1.0);
        mat[(1, 0)] = c(1.0);
        mat[(0, 1)] = c(0.5);
        mat[(1, 2)] = c(3.0);
        let mut out = vec![C64::new(0.0, 0.0); 3];
        row_mat_mul_into(&row, &mat, &mut out);
        assert_eq!(out[0], c(3.0));
        assert_eq!(out[1], c(0.5));
        assert_eq!(out[2], c(6.0));

        row_mat_mul_add(&row, &mat, &mut out);
        assert_eq!(out, vec![c(6.0), c(1.0), c(12.0)]);
    }

    #[test]
    fn test_averaged_row_mat_mul() {
        let new_row = vec![c(3.0), c(6.0)];
        let old_row = vec![c(1.0), c(2.0)];
        let mat = DMatrix::from_row_slice(2, 2, &[c(1.0), c(2.0), c(3.0), c(4.0)]);
        let mut out = vec![C64::new(0.0, 0.0); 2];
        averaged_row_mat_mul_into(&new_row, &old_row, &mat, &mut out);
        assert_eq!(out, vec![c(14.0), c(20.0)]);
    }

    #[test]
    fn test_add_assign() {
        let mut dst = vec![c(1.0), c(2.0)];
        let src = vec![c(3.0), c(4.0)];
        add_assign(&mut dst, &src);
        assert_eq!(dst, vec![c(4.0), c(6.0)]);
    }

    // ---------- fft / ifft ----------

    #[test]
    fn test_fft_ifft_roundtrip() {
        let nx = 8;
        let ncols = 3;
        let mut data = vec![0.0; nx * ncols];
        for col in 0..ncols {
            for i in 0..nx {
                data[idx(i, col, ncols)] = i as f64 + col as f64;
            }
        }
        let fwd = fft_along_x(&data, nx, ncols);
        let back = ifft_along_x(&fwd, nx, ncols);
        for (a, b) in data.iter().zip(back.iter()) {
            assert_close(*a, *b, 1e-12);
        }
    }

    #[test]
    fn test_fft_dc_component() {
        let nx = 8;
        let ncols = 1;
        let data = vec![3.0; nx];
        let fwd = fft_along_x(&data, nx, ncols);
        assert_close(fwd[0].re, 3.0 * nx as f64, 1e-12);
        assert_close(fwd[0].im, 0.0, 1e-12);
        for i in 1..nx {
            assert_close(fwd[i].re, 0.0, 1e-12);
            assert_close(fwd[i].im, 0.0, 1e-12);
        }
    }

    // ---------- nested2 ----------

    #[test]
    fn test_nested2() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = nested2(&data, 3, 2);
        assert_eq!(out, vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
    }

    // ---------- make_grid ----------

    #[test]
    fn test_make_grid_gill_shift() {
        let cfg = make_cfg(CaseKind::GillSym, 8, 6);
        let (x, y, z, dx, dy, dz) = make_grid(&cfg);
        assert_eq!(x.len(), 8);
        assert_eq!(y.len(), 7);
        assert_eq!(z.len(), 4);
        assert_close(dx, cfg.lx / 8.0, 1e-12);
        assert_close(dy, cfg.ly / 7.0, 1e-12);
        assert_close(dz, cfg.lz / 3.0, 1e-12);
        let expected_x0 = dx * (0.0 - 8.0 / 2.0) + 2.0 * PI;
        assert_close(x[0], expected_x0, 1e-12);
    }

    #[test]
    fn test_make_grid_kelvin_no_shift() {
        let cfg = make_cfg(CaseKind::KelvinWave, 8, 6);
        let (x, _, _, dx, _, _) = make_grid(&cfg);
        let expected_x0 = dx * (0.0 - 8.0 / 2.0);
        assert_close(x[0], expected_x0, 1e-12);
    }

    // ---------- initial_conditions ----------

    #[test]
    fn test_initial_conditions_gill_sym() {
        let cfg = make_cfg(CaseKind::GillSym, 16, 8);
        let (x, y, _, _, _, _) = make_grid(&cfg);
        let (v0, u0, eta0, q) = initial_conditions(&cfg, &x, &y);
        let nyp = cfg.ny + 1;
        assert_all_zero(&v0);
        assert_all_zero(&u0);
        assert_all_zero(&eta0);
        assert_nonzero(&q);
        assert!(q.iter().all(|&v| v >= 0.0));
        for i in 0..cfg.nx {
            for j in 0..nyp {
                assert_close(q[idx(i, j, nyp)], q[idx(i, cfg.ny - j, nyp)], 1e-12);
            }
        }
    }

    #[test]
    fn test_initial_conditions_gill_asym() {
        let cfg = make_cfg(CaseKind::GillAsym, 16, 8);
        let (x, y, _, _, _, _) = make_grid(&cfg);
        let (v0, u0, eta0, q) = initial_conditions(&cfg, &x, &y);
        let nyp = cfg.ny + 1;
        assert_all_zero(&v0);
        assert_all_zero(&u0);
        assert_all_zero(&eta0);
        assert_nonzero(&q);
        for i in 0..cfg.nx {
            for j in 0..nyp {
                assert_close(q[idx(i, j, nyp)], -q[idx(i, cfg.ny - j, nyp)], 1e-12);
            }
        }
    }

    #[test]
    fn test_initial_conditions_kelvin() {
        let cfg = make_cfg(CaseKind::KelvinWave, 16, 8);
        let (x, y, _, _, _, _) = make_grid(&cfg);
        let (v0, u0, eta0, q) = initial_conditions(&cfg, &x, &y);
        assert_all_zero(&v0);
        assert_all_zero(&q);
        assert_nonzero(&u0);
        assert_nonzero(&eta0);
        assert_eq!(u0.len(), eta0.len());
        for k in 0..u0.len() {
            assert_close(u0[k], eta0[k], 1e-12);
        }
    }

    #[test]
    fn test_initial_conditions_gauss_pert() {
        let cfg = make_cfg(CaseKind::GaussPert, 16, 8);
        let (x, y, _, _, _, _) = make_grid(&cfg);
        let (v0, u0, eta0, q) = initial_conditions(&cfg, &x, &y);
        assert_all_zero(&v0);
        assert_all_zero(&u0);
        assert_all_zero(&q);
        assert_nonzero(&eta0);
    }

    #[test]
    fn test_initial_conditions_matsuno() {
        let cfg = make_cfg(CaseKind::MatsunoHeating, 16, 8);
        let (x, y, _, _, _, _) = make_grid(&cfg);
        let (v0, u0, eta0, q) = initial_conditions(&cfg, &x, &y);
        assert_all_zero(&v0);
        assert_all_zero(&u0);
        assert_all_zero(&eta0);
        assert_nonzero(&q);
    }

    #[test]
    fn test_initial_conditions_gill_sym_gaussian_x() {
        let cfg = make_cfg(CaseKind::GillSymGaussianX, 16, 8);
        let (x, y, _, _, _, _) = make_grid(&cfg);
        let (v0, u0, eta0, q) = initial_conditions(&cfg, &x, &y);
        let nyp = cfg.ny + 1;
        assert_all_zero(&v0);
        assert_all_zero(&u0);
        assert_all_zero(&eta0);
        assert_nonzero(&q);
        for i in 0..cfg.nx {
            for j in 0..nyp {
                assert_close(q[idx(i, j, nyp)], q[idx(i, cfg.ny - j, nyp)], 1e-12);
            }
        }
    }

    #[test]
    fn test_initial_conditions_gill_sym_asym() {
        let cfg = make_cfg(CaseKind::GillSymAsym, 16, 8);
        let (x, y, _, _, _, _) = make_grid(&cfg);
        let (v0, u0, eta0, q) = initial_conditions(&cfg, &x, &y);
        assert_all_zero(&v0);
        assert_all_zero(&u0);
        assert_all_zero(&eta0);
        assert_nonzero(&q);
    }

    #[test]
    fn test_initial_conditions_gill_delta_y1() {
        let cfg = make_cfg(CaseKind::GillDeltaY1, 16, 8);
        let (x, y, _, _, _, _) = make_grid(&cfg);
        let (v0, u0, eta0, q) = initial_conditions(&cfg, &x, &y);
        assert_all_zero(&v0);
        assert_all_zero(&u0);
        assert_all_zero(&eta0);
        assert_nonzero(&q);
    }

    // ---------- v_to_eta_grid ----------

    #[test]
    fn test_v_to_eta_grid() {
        let nx = 2;
        let ny = 3;
        let v_raw = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let out = v_to_eta_grid(&v_raw, nx, ny);
        assert_eq!(out.len(), nx * (ny + 1));
        assert_close(out[idx(0, 0, ny + 1)], 0.5, 1e-12);
        assert_close(out[idx(0, 1, ny + 1)], 1.5, 1e-12);
        assert_close(out[idx(0, 3, ny + 1)], 1.5, 1e-12);
        assert_close(out[idx(1, 0, ny + 1)], 2.0, 1e-12);
        assert_close(out[idx(1, 1, ny + 1)], 4.5, 1e-12);
    }

    // ---------- CaseKind parse / label / short_name ----------

    #[test]
    fn test_case_parse_aliases() {
        let cases: &[(&str, &[&str])] = &[
            (
                "gill_sym",
                &["1", "gill_sym", "sym", "symmetric", "symmetric_heating"],
            ),
            (
                "gill_asym",
                &["2", "gill_asym", "asym", "asymmetric", "asymmetric_heating"],
            ),
            ("kelvin_wave", &["3", "kelvin", "kelvin_wave"]),
            (
                "gauss_pert",
                &["4", "gauss", "gaussian", "gauss_pert", "gaussian_initial"],
            ),
            (
                "matsuno_heating",
                &["5", "matsuno", "matsuno_heating", "forced_stationary"],
            ),
            (
                "gill_sym_gaussian_x",
                &[
                    "6",
                    "gill_sym_2",
                    "gill_sym_gaussian_x",
                    "symmetric_gaussian_x",
                ],
            ),
            (
                "gill_sym_asym",
                &["7", "gill_sym_asym", "sym_asym", "symmetric_asymmetric"],
            ),
            (
                "gill_delta_y1",
                &["8", "gill_delta_y1", "gill_sym_delta", "delta_y1"],
            ),
        ];
        for (canonical, aliases) in cases {
            let expected = CaseKind::parse(canonical).unwrap();
            for alias in *aliases {
                assert_eq!(CaseKind::parse(alias).unwrap(), expected);
            }
        }
    }

    #[test]
    fn test_case_parse_normalization() {
        assert_eq!(CaseKind::parse("  GILL-SYM  ").unwrap(), CaseKind::GillSym);
        assert_eq!(
            CaseKind::parse("Kelvin_Wave").unwrap(),
            CaseKind::KelvinWave
        );
    }

    #[test]
    fn test_case_parse_unknown() {
        let err = CaseKind::parse("nonsense").unwrap_err();
        pyo3::Python::attach(|py| {
            assert!(err.is_instance_of::<PyValueError>(py));
        });
    }

    #[test]
    fn test_kernel_parse_and_thread_limit() {
        assert_eq!(Kernel::parse("serial").unwrap(), Kernel::Serial);
        assert_eq!(Kernel::parse("  REYON ").unwrap(), Kernel::Reyon);
        assert!(Kernel::parse("rayon").is_err());
        assert_eq!(
            reyon_thread_pool().unwrap().current_num_threads(),
            MAX_RAYON_THREADS
        );
    }

    #[test]
    fn test_transform_cache_key_and_reuse() {
        let cfg = make_cfg(CaseKind::GillSym, 8, 4);
        let same_key_different_case = Config {
            case: CaseKind::KelvinWave,
            lz: cfg.lz * 2.0,
            nz: cfg.nz + 2,
            ..cfg.clone()
        };
        assert_eq!(
            TransformCacheKey::from_config(&cfg),
            TransformCacheKey::from_config(&same_key_different_case)
        );

        let mut different_timestep = cfg.clone();
        different_timestep.nt *= 2;
        assert_ne!(
            TransformCacheKey::from_config(&cfg),
            TransformCacheKey::from_config(&different_timestep)
        );

        let first = transforms_for(&cfg).unwrap();
        let second = transforms_for(&cfg).unwrap();
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn test_case_labels() {
        assert_eq!(CaseKind::GillSym.label(), "Gill: symmetric heating");
        assert_eq!(CaseKind::GillAsym.label(), "Gill: asymmetric heating");
        assert_eq!(CaseKind::KelvinWave.label(), "Kelvin wave");
        assert_eq!(
            CaseKind::GaussPert.label(),
            "Gaussian-shape initial perturbation"
        );
        assert_eq!(
            CaseKind::MatsunoHeating.label(),
            "Matsuno: forced stationary motion"
        );
        assert_eq!(
            CaseKind::GillSymGaussianX.label(),
            "Gill: symmetric heating with Gaussian shape in zonal direction"
        );
        assert_eq!(
            CaseKind::GillSymAsym.label(),
            "Gill: symmetric heating + asymmetric heating"
        );
        assert_eq!(
            CaseKind::GillDeltaY1.label(),
            "Gill: delta heating at y = 1"
        );
    }

    #[test]
    fn test_case_short_names() {
        assert_eq!(CaseKind::GillSym.short_name(), "Gill_sym");
        assert_eq!(CaseKind::GillAsym.short_name(), "Gill_asym");
        assert_eq!(CaseKind::KelvinWave.short_name(), "KelvinWave");
        assert_eq!(CaseKind::GaussPert.short_name(), "GaussPert");
        assert_eq!(CaseKind::MatsunoHeating.short_name(), "MatsunoHeating");
        assert_eq!(CaseKind::GillSymGaussianX.short_name(), "Gill_sym_2");
        assert_eq!(CaseKind::GillSymAsym.short_name(), "Gill_sym_asym");
        assert_eq!(CaseKind::GillDeltaY1.short_name(), "Gill_sym_delta");
    }

    // ---------- build_config ----------

    #[test]
    fn test_build_config_defaults() {
        let cfg =
            build_config("gill_sym", None, None, 128, 60, 20, None, None, None, None).unwrap();
        assert_eq!(cfg.t, 50.0);
        assert_eq!(cfg.nt, 500);
        assert_close(cfg.lx, 12.0 * PI, 1e-12);
        assert_close(cfg.ly, 5.0 * PI, 1e-12);
        assert_close(cfg.lz, PI, 1e-12);
        assert_eq!(cfg.damping, 0.1);
    }

    #[test]
    fn test_build_config_case_overrides() {
        let cfg = build_config(
            "kelvin_wave",
            None,
            None,
            128,
            60,
            20,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(cfg.t, 10.0);
        assert_eq!(cfg.nt, 100);
        assert_eq!(cfg.damping, 0.0);

        let cfg = build_config(
            "gauss_pert",
            None,
            None,
            128,
            60,
            20,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(cfg.damping, 0.0);

        let cfg = build_config(
            "matsuno_heating",
            None,
            None,
            128,
            60,
            20,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(cfg.damping, 0.2);
    }

    #[test]
    fn test_build_config_explicit_overrides() {
        let cfg = build_config(
            "kelvin_wave",
            Some(20.0),
            Some(200),
            64,
            30,
            10,
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(0.5),
        )
        .unwrap();
        assert_eq!(cfg.t, 20.0);
        assert_eq!(cfg.nt, 200);
        assert_eq!(cfg.nx, 64);
        assert_eq!(cfg.ny, 30);
        assert_eq!(cfg.nz, 10);
        assert_eq!(cfg.lx, 1.0);
        assert_eq!(cfg.ly, 2.0);
        assert_eq!(cfg.lz, 3.0);
        assert_eq!(cfg.damping, 0.5);
    }

    #[test]
    fn test_build_config_validation_errors() {
        for bad in [
            build_config(
                "gill_sym",
                None,
                Some(0),
                128,
                60,
                20,
                None,
                None,
                None,
                None,
            ),
            build_config("gill_sym", None, None, 0, 60, 20, None, None, None, None),
            build_config("gill_sym", None, None, 128, 0, 20, None, None, None, None),
            build_config("gill_sym", None, None, 128, 60, 1, None, None, None, None),
        ] {
            let err = bad.unwrap_err();
            pyo3::Python::attach(|py| {
                assert!(err.is_instance_of::<PyValueError>(py));
            });
        }
    }

    // ---------- simulate (end-to-end) ----------

    #[test]
    fn test_simulate_all_cases_finite_and_shapes() {
        let cases = [
            "gill_sym",
            "gill_asym",
            "kelvin_wave",
            "gauss_pert",
            "matsuno_heating",
            "gill_sym_gaussian_x",
            "gill_sym_asym",
            "gill_delta_y1",
        ];
        let nx = 16;
        let ny = 8;
        let nz = 4;
        let nt = 10;
        for case in cases {
            pyo3::Python::attach(|py| {
                let result = gill_simulate(
                    py,
                    case,
                    None,
                    Some(nt),
                    nx,
                    ny,
                    nz,
                    None,
                    None,
                    None,
                    None,
                    0,
                    "serial",
                )
                .unwrap();
                let bound = result.bind(py);

                let case_out: String = bound.get_item("case").unwrap().unwrap().extract().unwrap();
                assert_eq!(case_out.as_str(), case);

                let nx_out: usize = bound.get_item("nx").unwrap().unwrap().extract().unwrap();
                let ny_out: usize = bound.get_item("ny").unwrap().unwrap().extract().unwrap();
                let nz_out: usize = bound.get_item("nz").unwrap().unwrap().extract().unwrap();
                assert_eq!(nx_out, nx);
                assert_eq!(ny_out, ny);
                assert_eq!(nz_out, nz);

                let z: Vec<f64> = bound.get_item("z").unwrap().unwrap().extract().unwrap();
                assert_eq!(z.len(), nz);

                let eta: Vec<Vec<f64>> = bound.get_item("eta").unwrap().unwrap().extract().unwrap();
                let u: Vec<Vec<f64>> = bound.get_item("u").unwrap().unwrap().extract().unwrap();
                let v: Vec<Vec<f64>> = bound.get_item("v").unwrap().unwrap().extract().unwrap();
                let q: Vec<Vec<f64>> = bound.get_item("q").unwrap().unwrap().extract().unwrap();
                let v_staggered: Vec<Vec<f64>> = bound
                    .get_item("v_staggered")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();

                assert_eq!(eta.len(), nx);
                assert_eq!(eta[0].len(), ny + 1);
                assert_eq!(u.len(), nx);
                assert_eq!(u[0].len(), ny + 1);
                assert_eq!(v.len(), nx);
                assert_eq!(v[0].len(), ny + 1);
                assert_eq!(q.len(), nx);
                assert_eq!(q[0].len(), ny + 1);
                assert_eq!(v_staggered.len(), nx);
                assert_eq!(v_staggered[0].len(), ny);

                for row in eta
                    .iter()
                    .chain(u.iter())
                    .chain(v.iter())
                    .chain(q.iter())
                    .chain(v_staggered.iter())
                {
                    for &val in row {
                        assert!(val.is_finite(), "non-finite value in case {case}");
                    }
                }
            });
        }
    }

    #[test]
    fn test_simulate_gill_sym_q_symmetric() {
        let nx = 16;
        let ny = 8;
        let nz = 4;
        let nt = 10;
        pyo3::Python::attach(|py| {
            let result = gill_simulate(
                py,
                "gill_sym",
                None,
                Some(nt),
                nx,
                ny,
                nz,
                None,
                None,
                None,
                None,
                0,
                "serial",
            )
            .unwrap();
            let bound = result.bind(py);
            let q: Vec<Vec<f64>> = bound.get_item("q").unwrap().unwrap().extract().unwrap();
            for i in 0..nx {
                for j in 0..(ny + 1) {
                    assert_close(q[i][j], q[i][ny - j], 1e-12);
                }
            }
        });
    }

    #[test]
    fn test_simulate_q_matches_initial_conditions() {
        let nx = 16;
        let ny = 8;
        let nz = 4;
        let nt = 5;
        pyo3::Python::attach(|py| {
            let result = gill_simulate(
                py,
                "gill_sym",
                None,
                Some(nt),
                nx,
                ny,
                nz,
                None,
                None,
                None,
                None,
                0,
                "serial",
            )
            .unwrap();
            let bound = result.bind(py);
            let q_sim: Vec<Vec<f64>> = bound.get_item("q").unwrap().unwrap().extract().unwrap();

            let cfg = build_config(
                "gill_sym",
                None,
                Some(nt),
                nx,
                ny,
                nz,
                None,
                None,
                None,
                None,
            )
            .unwrap();
            let (x, y, _, _, _, _) = make_grid(&cfg);
            let (_, _, _, q) = initial_conditions(&cfg, &x, &y);
            let nyp = ny + 1;
            for i in 0..nx {
                for j in 0..nyp {
                    assert_close(q_sim[i][j], q[idx(i, j, nyp)], 1e-12);
                }
            }
        });
    }

    #[test]
    fn test_simulate_small_grid_reference_values() {
        pyo3::Python::attach(|py| {
            let result = gill_simulate(
                py,
                "gill_sym",
                Some(0.3),
                Some(3),
                8,
                4,
                3,
                None,
                None,
                None,
                None,
                0,
                "serial",
            )
            .unwrap();
            let result = result.bind(py);
            let eta: Vec<Vec<f64>> = result.get_item("eta").unwrap().unwrap().extract().unwrap();
            let u: Vec<Vec<f64>> = result.get_item("u").unwrap().unwrap().extract().unwrap();
            let v: Vec<Vec<f64>> = result
                .get_item("v_staggered")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();

            for (actual, expected) in [
                (eta[0][0], 3.259977822981786e-7),
                (eta[3][2], -0.09701921395082694),
                (eta[7][4], -2.618887749877964e-8),
                (u[0][0], -4.1865110366568964e-7),
                (u[3][2], 6.604212157133154e-8),
                (u[7][4], 1.439953554891539e-8),
                (v[0][0], 3.4987038050159007e-6),
                (v[3][2], -0.0041991114113769665),
                (v[7][3], 7.220311762872302e-8),
            ] {
                assert_close(actual, expected, 1e-14);
            }
        });
    }

    #[test]
    fn test_simulate_all_cases_reference_checksums() {
        let cases = [
            ("gill_sym", -89.6147441302599, -193.73048135439132),
            ("gill_asym", -44.56465214837573, -89.16553665605649),
            ("kelvin_wave", -779.4960681128259, -2363.9877402156003),
            ("gauss_pert", 1075.2029233962191, 3260.135153787091),
            ("matsuno_heating", -11.584156847697127, -28.828623414621234),
            (
                "gill_sym_gaussian_x",
                -80.7305155251585,
                -174.55031942260655,
            ),
            ("gill_sym_asym", -104.38674896376692, -223.26307336727234),
            ("gill_delta_y1", -918.3589598722483, -1972.2704706795823),
        ];

        for (case, expected_final, expected_history) in cases {
            pyo3::Python::attach(|py| {
                let result = gill_simulate(
                    py,
                    case,
                    Some(0.3),
                    Some(3),
                    8,
                    12,
                    3,
                    None,
                    None,
                    None,
                    None,
                    1,
                    "serial",
                )
                .unwrap();
                let result = result.bind(py);
                let eta: Vec<Vec<f64>> =
                    result.get_item("eta").unwrap().unwrap().extract().unwrap();
                let u: Vec<Vec<f64>> = result.get_item("u").unwrap().unwrap().extract().unwrap();
                let v: Vec<Vec<f64>> = result
                    .get_item("v_staggered")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();
                let eta_history: Vec<Vec<Vec<f64>>> = result
                    .get_item("eta_history")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();
                let u_history: Vec<Vec<Vec<f64>>> = result
                    .get_item("u_history")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();
                let v_history: Vec<Vec<Vec<f64>>> = result
                    .get_item("v_history")
                    .unwrap()
                    .unwrap()
                    .extract()
                    .unwrap();

                let final_checksum =
                    weighted_2d(&eta) + 3.0 * weighted_2d(&u) + 5.0 * weighted_2d(&v);
                let history_checksum = weighted_3d(&eta_history)
                    + 3.0 * weighted_3d(&u_history)
                    + 5.0 * weighted_3d(&v_history);
                assert_close(final_checksum, expected_final, 1e-10);
                assert_close(history_checksum, expected_history, 1e-10);
            });
        }
    }

    #[test]
    fn test_simulate_store_every() {
        let nx = 16;
        let ny = 8;
        let nz = 4;
        let nt = 10;
        pyo3::Python::attach(|py| {
            let result = gill_simulate(
                py,
                "gill_sym",
                None,
                Some(nt),
                nx,
                ny,
                nz,
                None,
                None,
                None,
                None,
                5,
                "serial",
            )
            .unwrap();
            let bound = result.bind(py);
            let time: Vec<f64> = bound.get_item("time").unwrap().unwrap().extract().unwrap();
            let eta_history: Vec<Vec<Vec<f64>>> = bound
                .get_item("eta_history")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let u_history: Vec<Vec<Vec<f64>>> = bound
                .get_item("u_history")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            let v_history: Vec<Vec<Vec<f64>>> = bound
                .get_item("v_history")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(time.len(), 2);
            assert_eq!(eta_history.len(), 2);
            assert_eq!(u_history.len(), 2);
            assert_eq!(v_history.len(), 2);
            assert_eq!(eta_history[0].len(), nx);
            assert_eq!(eta_history[0][0].len(), ny + 1);
        });
    }

    #[test]
    fn test_simulate_no_store() {
        let nx = 16;
        let ny = 8;
        let nz = 4;
        let nt = 10;
        pyo3::Python::attach(|py| {
            let result = gill_simulate(
                py,
                "gill_sym",
                None,
                Some(nt),
                nx,
                ny,
                nz,
                None,
                None,
                None,
                None,
                0,
                "serial",
            )
            .unwrap();
            let bound = result.bind(py);
            let time: Vec<f64> = bound.get_item("time").unwrap().unwrap().extract().unwrap();
            let eta_history: Vec<Vec<Vec<f64>>> = bound
                .get_item("eta_history")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert!(time.is_empty());
            assert!(eta_history.is_empty());
        });
    }

    #[test]
    fn test_simulate_reyon_matches_serial() {
        pyo3::Python::attach(|py| {
            let serial = gill_simulate(
                py,
                "gill_sym",
                None,
                Some(8),
                16,
                8,
                4,
                None,
                None,
                None,
                None,
                0,
                "serial",
            )
            .unwrap();
            let reyon = gill_simulate(
                py,
                "gill_sym",
                None,
                Some(8),
                16,
                8,
                4,
                None,
                None,
                None,
                None,
                0,
                "reyon",
            )
            .unwrap();

            let serial = serial.bind(py);
            let reyon = reyon.bind(py);
            let reyon_kernel: String = reyon
                .get_item("kernel")
                .unwrap()
                .unwrap()
                .extract()
                .unwrap();
            assert_eq!(reyon_kernel, "reyon");

            for field in ["eta", "u", "v", "v_staggered"] {
                let serial_field: Vec<Vec<f64>> =
                    serial.get_item(field).unwrap().unwrap().extract().unwrap();
                let reyon_field: Vec<Vec<f64>> =
                    reyon.get_item(field).unwrap().unwrap().extract().unwrap();
                assert_eq!(
                    serial_field, reyon_field,
                    "parallel result differs for {field}"
                );
            }
        });
    }
}
