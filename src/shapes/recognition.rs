use crate::shapes::similarity::{unit_sphere_normalize, polyhedron};
use crate::shapes::{shape_from_name, Name, Matrix3N, SHAPES};

use itertools::Itertools;
use rand::distributions::Distribution;
use statrs::distribution::{Continuous, ContinuousCDF, Normal, Beta, ChiSquared};
use statrs::statistics::Statistics;

pub const CSM_RESCALE: f64 = 150.0;

pub fn sample(s: Name) -> f64 {
    let n = shape_from_name(s).size();

    let norm_distribution = Normal::new(1.0, 0.2).unwrap();
    let mut prng = rand::thread_rng();

    let mut cloud = Matrix3N::new_random(n);
    for mut v in cloud.column_iter_mut() {
        let factor = norm_distribution.sample(&mut prng) / v.norm();
        v *= factor;
    }

    // add centroid and normalize
    cloud = unit_sphere_normalize(cloud.insert_column(n, 0.0));

    polyhedron(cloud, s).expect("No algorithm failures").csm
}

fn beta_parameters(samples: &Vec<f64>) -> (f64, f64) {
    let mean = samples.mean();
    let variance = samples.variance();

    let alpha = mean * (mean * (1.0 - mean) / variance - 1.0);
    let beta = (1.0 - mean) * (mean * (1.0 - mean) / variance - 1.0);

    (alpha, beta)
}

pub fn random_cloud_distribution(s: Name, n: usize) -> Beta {
    let mut samples = vec![f64::default(); n];
    for v in samples.iter_mut() {
        *v = sample(s) / CSM_RESCALE;
    }

    assert!(samples.as_slice().max() <= 1.0);
    let (alpha, beta) = beta_parameters(&samples);
    Beta::new(alpha, beta).unwrap()
}

pub fn embedded_distribution(s: Name) -> Beta {
    match s {
        Name::Line => Beta::new(3.7709, 19.3418).unwrap(),
        Name::Bent => Beta::new(1.6006, 32.0930).unwrap(),
        Name::EquilateralTriangle => Beta::new(5.3401, 32.7830).unwrap(),
        Name::VacantTetrahedron => Beta::new(4.3195, 34.6502).unwrap(),
        Name::T => Beta::new(5.9063, 52.1758).unwrap(),
        Name::Tetrahedron => Beta::new(7.3400, 34.4143).unwrap(),
        Name::Square => Beta::new(11.8430, 53.6998).unwrap(),
        Name::Seesaw => Beta::new(8.2791, 52.1629).unwrap(),
        Name::TrigonalPyramid => Beta::new(5.9750, 36.2147).unwrap(),
        Name::SquarePyramid => Beta::new(12.0397, 61.0895).unwrap(),
        Name::TrigonalBipyramid => Beta::new(10.5388, 51.2083).unwrap(),
        Name::Pentagon => Beta::new(19.9409, 77.9127).unwrap(),
        Name::Octahedron => Beta::new(19.5386, 75.2430).unwrap(),
        Name::TrigonalPrism => Beta::new(17.1139, 79.9028).unwrap(),
        Name::PentagonalPyramid => Beta::new(18.9412, 89.5201).unwrap(),
        Name::Hexagon => Beta::new(21.8653, 79.3234).unwrap(),
        Name::PentagonalBipyramid => Beta::new(21.0180, 90.9769).unwrap(),
        Name::CappedOctahedron => Beta::new(18.6539, 85.6336).unwrap(),
        Name::CappedTrigonalPrism => Beta::new(17.0386, 80.1077).unwrap(),
        Name::SquareAntiprism => Beta::new(26.3346, 112.8372).unwrap(),
        Name::Cube => Beta::new(27.4242, 109.7715).unwrap(),
        Name::TrigonalDodecahedron => Beta::new(22.9751, 101.4186).unwrap(),
        Name::HexagonalBipyramid => Beta::new(27.4513, 115.6199).unwrap(),
        Name::TricappedTrigonalPrism => Beta::new(27.1751, 115.7056).unwrap(),
        Name::CappedSquareAntiprism => Beta::new(28.4785, 121.4221).unwrap(),
        Name::HeptagonalBipyramid => Beta::new(28.0546, 119.8896).unwrap(),
        Name::BicappedSquareAntiprism => Beta::new(30.0449, 129.6707).unwrap(),
        Name::EdgeContractedIcosahedron => Beta::new(33.5518, 149.9625).unwrap(),
        Name::Icosahedron => Beta::new(38.9508, 158.8088).unwrap(),
        Name::Cuboctahedron => Beta::new(34.9043, 148.4051).unwrap(),
    }
}

pub fn reduced_chi_squared(samples: &[f64], beta: Beta) -> f64 {
    assert!(samples.len() >= 10);

    const QUANTILE: f64 = 0.1;

    // Histogram bin and sum

    let n_bins = (samples.len() / 10).max(10);
    let (sample_min, sample_max) = match samples.iter().minmax() {
        itertools::MinMaxResult::MinMax(a, b) => (a, b + 1e-6),
        _ => panic!("Expected at least two values in samples")
    };
    let pdf_min = beta.cdf(QUANTILE);
    let pdf_max = beta.cdf(1.0 - QUANTILE);
    let min = sample_min.min(pdf_min);
    let max = sample_max.max(pdf_max);
    let h = (max - min) / n_bins as f64;

    let mut chi_squared = 0.0;

    for i in 0..n_bins {
        let lower = min + i as f64 * h;
        let upper = lower + h;

        let expected_count = (beta.cdf(upper) - beta.cdf(lower)) * samples.len() as f64;
        let observed_count = samples.iter().filter(|&&v| lower <= v && v < upper).count() as f64;

        if expected_count == 0.0 && observed_count > 0.0 {
            panic!("Encountered unexpected counts in bin");
        }

        if expected_count == 0.0 && observed_count == 0.0 {
            // Skip empty bins
            continue;
        }

        chi_squared += (observed_count - expected_count).powi(2) / expected_count;
    }

    // Beta distribution has two parameters, plus one more
    let dof = samples.len() - 3;

    chi_squared / dof as f64
}

pub fn goodness_of_fit(samples: &[f64], beta: Beta) -> f64 {
    ChiSquared::new(2.0).unwrap().pdf(reduced_chi_squared(samples, beta))
}

pub fn least_likely_random(cloud: &Matrix3N) -> (Name, f64) {
    let likelihood_random = |name: Name| {
        let csm = polyhedron(cloud.clone(), name).expect("Similarities worked fine").csm;
        embedded_distribution(name).cdf(csm / CSM_RESCALE)
    };

    SHAPES.iter()
        .filter(|s| s.size() + 1 == cloud.ncols())
        .map(|s| (s.name, likelihood_random(s.name)))
        .min_by(|(_, p), (_, q)| p.partial_cmp(q).expect("No NaNs from similarities"))
        .expect("At least one fitting shape size")
}

#[cfg(test)]
mod tests {

}
