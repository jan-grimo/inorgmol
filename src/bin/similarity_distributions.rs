use molassembler::shapes::recognition::{random_cloud_distribution, CSM_RESCALE};
use molassembler::shapes::{Name, SHAPES};

use csv::Writer;
use std::error::Error;
use statrs::statistics::Statistics;

const SAMPLES: usize = 100;

enum Mode {
    ConvergenceMetrics,
    ShapeDistributionParameters
}

const MODE: Mode = Mode::ShapeDistributionParameters;

fn write_convergence_metrics(name: Name) -> Result<(), Box<dyn Error>> {
    let mut samples = vec![f64::default(); SAMPLES];
    for v in samples.iter_mut() {
        *v = molassembler::shapes::recognition::sample(name) / CSM_RESCALE;
    }

    let mut writer = Writer::from_path("samples.csv")?;
    for n in 1..=SAMPLES {
        let sample = samples[n-1] * CSM_RESCALE;
        let mean = samples[0..n].mean();
        let variance = samples[0..n].variance();

        let alpha = mean * (mean * (1.0 - mean) / variance - 1.0);
        let beta = (1.0 - mean) * (mean * (1.0 - mean) / variance - 1.0);

        let numbers = [sample, mean, variance, alpha, beta];
        let strings = numbers.map(|v| v.to_string());

        writer.write_record(&strings)?;
    }

    writer.flush()?;
    Ok(())
}

fn shape_distribution_parameters() -> Result<(), Box<dyn Error>> {
    for shape in SHAPES.iter() {
        let distr = random_cloud_distribution(shape.name, 500);
        println!("{} => Beta::new({:.4}, {:.4}),", shape.name, distr.shape_a(), distr.shape_b());
    }

    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    match MODE {
        Mode::ConvergenceMetrics => write_convergence_metrics(Name::Tetrahedron),
        Mode::ShapeDistributionParameters => shape_distribution_parameters()
    }
}
