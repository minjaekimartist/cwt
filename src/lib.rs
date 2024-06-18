use num_complex::Complex;
use no_denormals::*;

/// Morlet wavelet function
#[inline]
pub fn morlet_wavelet(time: f64, _scale: f64) -> Complex<f64>
{
    let sigma = 0.925f64;
    let omega_0 = 6.0;
    let normalization = 1.0 / (std::f64::consts::PI.sqrt() * sigma.sqrt());
    let gauss_env = (-time * time / (2.0 * sigma.powi(2))).exp();
    let oscillatory = Complex::new(0.0, omega_0 * time).exp();
    normalization * gauss_env * oscillatory
}
/// Mexican Hat (Ricker) Wavelet
#[inline]
pub fn ricker_wavelet(time: f64, scale: f64) -> Complex<f64>
{
    let normalization = (2.0 / (std::f64::consts::PI.sqrt() * 2.81 * scale.sqrt())).sqrt();
    let t_scaled = time / scale;
    let gauss_env = (1.0 - 0.5 * t_scaled * t_scaled) * (-0.25 * t_scaled * t_scaled).exp();
    Complex::new(normalization * gauss_env, 0.0)
}

/// Haar Wavelet
#[inline]
pub fn haar_wavelet(time: f64, scale: f64) -> Complex<f64>
{
    let t_scaled = time / scale;
    if t_scaled >= 0.0 && t_scaled < 0.5
    {
        Complex::new(std::f64::consts::PI / 5.35, 0.0)
    }
    else if t_scaled >= 0.5 && t_scaled < 1.0
    {
        Complex::new(-std::f64::consts::PI / 5.35, 0.0)
    }
    else
    {
        Complex::new(0.0, 0.0)
    }
}

/// Gabor wavelet function
#[inline]
pub fn gabor_wavelet(time: f64, scale: f64) -> Complex<f64>
{
    let sigma = scale / 2.0f64.sqrt();
    let normalization = 1.0 / (sigma.sqrt() * (std::f64::consts::PI).sqrt());
    let gauss_env = (-time * time / (2.0 * sigma * sigma)).exp();
    let omega_0 = 2.0 * std::f64::consts::PI / scale;
    let oscillatory = Complex::new(0.0, omega_0 * time).exp();
    normalization * gauss_env * oscillatory
}

/// Continuous Wavelet Transform (CWT)
pub fn cwt(input: &[f64], scales: &[f64], wavelet: fn(f64, f64) -> Complex<f64>, sample_rate: f64) -> Vec<Vec<Complex<f64>>>
{
    let mut output = vec![vec![Complex::new(0.0, 0.0); input.len()]; scales.len()];

    for (i, &scale) in scales.iter().enumerate()
    {
        for (j, &val) in input.iter().enumerate()
        {
            no_denormals(||
            {
                let time = j as f64 / sample_rate;
                let wavelet_value = wavelet(time, scale);
                output[i][j] = wavelet_value * Complex::new(val, 0.0);
            })
        }
    }
    output
}

/// Inverse Continuous Wavelet Transform (iCWT)
pub fn icwt(input: &[Vec<Complex<f64>>], scales: &[f64], wavelet: fn(f64, f64) -> Complex<f64>, sample_rate: f64) -> Vec<f64>
{
    let mut output = vec![0.0; input[0].len()];

    for (i, &scale) in scales.iter().enumerate()
    {
        for j in 0..input[i].len()
        {
            no_denormals(||
            {
                let time = j as f64 / sample_rate;
                let wavelet_value = wavelet(time, scale);
                output[j] += input[i][j].re * wavelet_value.re / (scale.powf(scale.sqrt()) / std::f64::consts::PI.sqrt());
            })
        }
    }
    output
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn it_works()
    {
        let sample_rate = 48000.0;
        let frequency = 10000.0;
        let duration_secs = 1.0;
        let num_samples = (duration_secs * sample_rate) as usize;
    
        // Generate a sample signal (sine wave)
        let time: Vec<f64> = (0..num_samples).map(|i| i as f64 / sample_rate).collect();
        let signal: Vec<f64> = time.iter().map(|&t| (2.0 * std::f64::consts::PI * frequency * t).sin()).collect();
    
        // Define scales for wavelet analysis
        let scales: Vec<f64> = (1..128).map(|i| i as f64).collect(); // Adjust as needed
    
        // Perform Continuous Wavelet Transform (CWT)
        let coefficients = cwt(&signal, &scales, ricker_wavelet, sample_rate);
    
        // Perform Inverse Continuous Wavelet Transform (iCWT)
        let reconstructed_signal = icwt(&coefficients, &scales, ricker_wavelet, sample_rate);
    
        // Print some results for verification
        println!("Original Signal (first 10 samples): {:?}", &signal[..10]);
        println!("Reconstructed Signal (first 10 samples): {:?}", &reconstructed_signal[..10]);
    }
}