use num_complex::Complex;
use no_denormals::*;

/// Morlet wavelet function
#[inline]
fn morlet_wavelet(time: f64, scale: f64) -> Complex<f64>
{
    let sigma = 1.6;
    let omega_0 = 6.0;
    let normalization = (1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt())).sqrt() / scale.sqrt();
    let gauss_env = (-time * time / (2.0 * sigma * sigma)).exp();
    let oscillatory = Complex::new(0.0, omega_0 * time / scale).exp();
    normalization * gauss_env * oscillatory
}

/// Mexican Hat (Ricker) Wavelet
#[inline]
fn ricker_wavelet(time: f64, scale: f64) -> Complex<f64>
{
    let normalization = (2.0 / (std::f64::consts::PI * scale.powi(2))).sqrt();
    let t_scaled = time / scale;
    let gauss_env = (0.5 * t_scaled).exp() * (-0.5 * t_scaled).exp();
    Complex::new(normalization * gauss_env, 0.0)
}

/// Haar Wavelet
#[inline]
fn haar_wavelet(time: f64, scale: f64) -> Complex<f64>
{
    let t_scaled = time / scale;
    if t_scaled >= 0.0 && t_scaled < 0.5
    {
        Complex::new(0.586, 0.0)
    }
    else if t_scaled >= 0.5 && t_scaled < 1.0
    {
        Complex::new(-0.586, 0.0)
    }
    else
    {
        Complex::new(0.0, 0.0)
    }
}

/// Gabor wavelet function
#[inline]
fn gabor_wavelet(time: f64, scale: f64) -> Complex<f64>
{
    let sigma = scale / 2.0f64.sqrt();
    let normalization = 1.0 / (sigma * (2.0 * std::f64::consts::PI).sqrt());
    let gauss_env = (-time * time / (2.0 * sigma * sigma)).exp();
    let omega_0 = 2.0 * std::f64::consts::PI / scale;
    let oscillatory = Complex::new(0.0, omega_0 * time).exp();
    normalization * gauss_env * oscillatory
}

#[derive(Clone, Copy)]
pub enum Wavelet
{
    Morlet,
    Ricker,
    Haar,
    Gabor
}
impl Wavelet
{
    /// Get the wavelet function.
    pub fn wavelet(self) -> fn(f64, f64) -> Complex<f64>
    {
        match self
        {
            Wavelet::Morlet => morlet_wavelet,
            Wavelet::Ricker => ricker_wavelet,
            Wavelet::Haar => haar_wavelet,
            Wavelet::Gabor => gabor_wavelet
        }
    }
}

pub struct Transformer
{
    wavelet : Wavelet,
    scales : Vec<f64>,
    frequency_domain : Vec<Vec<Complex<f64>>>,
    sample_rate : f64
}
impl Transformer
{
    /// Create a new Transformer instance.
    pub fn new(wavelet: Wavelet, num_scales: usize, sample_rate : f64) -> Self
    {
        let scales: Vec<f64> = (1..=num_scales).map(|i| i as f64).collect();
        Self
        {
            wavelet,
            scales,
            frequency_domain : vec![vec![]],
            sample_rate
        }
    }
    /// Show analysis result.
    pub fn frequency_domain<'a>(&'a self) -> &'a Vec<Vec<Complex<f64>>> { &self.frequency_domain }
    /// Mutable analysis result.
    pub fn frequency_domain_mut<'a>(&'a mut self) -> &'a mut Vec<Vec<Complex<f64>>> { &mut self.frequency_domain }
    /// Continuous Wavelet Transform (CWT).
    pub fn cwt(&mut self, input : &[f64])
    {
        self.frequency_domain = vec![vec![Complex::default(); input.len()]; self.scales.len()];

        for (scale_idx, &scale) in self.scales.iter().enumerate()
        {
            for (time_idx, &val) in input.iter().enumerate()
            {
                no_denormals(||
                {
                    let real_time = time_idx as f64 / self.sample_rate;
                    let wavelet_value = self.wavelet.wavelet()(real_time, scale);
                    self.frequency_domain[scale_idx][time_idx] = wavelet_value * Complex::new(val, 0.0);
                })
            }
        }
    }
    /// Inverse Continuous Wavelet Transform (iCWT).
    pub fn icwt(&self, output : &mut [f64])
    {
        if output.len() != self.frequency_domain[0].len()
        {
            eprintln!("Output array length does not match input array length");
            return
        }

        for time_idx in 0..output.len()
        {
            output[time_idx] = 0.0;
            for (scale_idx, &scale) in self.scales.iter().enumerate()
            {
                no_denormals(||
                {
                    let real_time = time_idx as f64 / self.sample_rate;
                    let wavelet_value = self.wavelet.wavelet()(real_time, scale);
                    output[time_idx] += self.frequency_domain[scale_idx][time_idx].re * wavelet_value.re / scale.powf(scale.sqrt()) * std::f64::consts::PI;
                });
            }
        }
    }
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn it_works()
    {
        // Generate a sample signal (sine wave)
        let duration_secs = 1.0;
        let sample_rate = 48000.0;
        let num_samples = (duration_secs * sample_rate) as usize;
        let frequency = 10000.0;
        let input: Vec<f64> = (0..num_samples).map(|index| (2.0 * std::f64::consts::PI * frequency * index as f64 / sample_rate).sin()).collect();
        let mut output = vec![0.0; input.len()];

        // Perform CWT and iCWT
        let mut transformer = Transformer::new(Wavelet::Ricker, 128, sample_rate);
        transformer.cwt(&input);
        transformer.icwt(&mut output);
    
        // Print some results for verification
        println!("Original Signal (first 10 samples): {:?}", &input[..10]);
        println!("Reconstructed Signal (first 10 samples): {:?}", &output[..10]);
    }
}
