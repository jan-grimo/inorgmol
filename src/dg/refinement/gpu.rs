use crate::dg::DIMENSIONS;
use crate::dg::refinement::{Stage, Bounds, DistanceBound, RefinementErrorFunction, Parallel, four_mut};
use std::borrow::Cow;
use wgpu::util::DeviceExt;

use crate::dg::refinement::Float;

extern crate nalgebra as na;

// IDEAS
// - Could try calculating numerical hessian on GPU, since we know which terms
//   impact which parameters, maybe possible to make it nice and efficient

struct GpuHandles {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
}

impl GpuHandles {
    async fn construct() -> Option<GpuHandles> {
        let adapter = wgpu::Instance::default()
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::downlevel_defaults(),
                },
                None,
            )
            .await
            .ok()?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Distance gradient"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("dist_grad.wgsl"))),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: "main",
        });

        Some(GpuHandles {device, queue, pipeline})
    }

    fn new() -> Option<GpuHandles> {
        pollster::block_on(Self::construct())
    }

    async fn distance_gradient_contributions<F: Float>(&self, positions: &na::DVector<F>, gpu_linear_squared_bounds: &na::Matrix2xX<f32>) -> na::DVector<F> {
        let n = positions.len() / 4;
        // TODO this is an unnecessary copy if F is f32
        let f32_positions = na::DVector::<f32>::from_iterator(positions.len(), positions.iter().map(|f| f.to_f32().unwrap()));

        let linear_length = (n.pow(2) - n) / 2;
        let linear_slice_size = linear_length * 4 * std::mem::size_of::<f32>();
        let linear_size = linear_slice_size as wgpu::BufferAddress;

        let positions_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Positions buffer"),
            contents: bytemuck::cast_slice(f32_positions.data.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let bounds_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Bounds buffer"),
            contents: bytemuck::cast_slice(gpu_linear_squared_bounds.data.as_slice()),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let gradient_contribution_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gradient buffer"),
            size: linear_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Distance gradient buffers"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bounds_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gradient_contribution_buffer.as_entire_binding(),
                },
            ],
        });

        let gradient_contribution_return_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Gradient return buffer"),
            size: linear_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.insert_debug_marker("Compute distance gradient contributions");
            cpass.dispatch_workgroups(n as u32, n as u32, 1);
        }

        encoder.copy_buffer_to_buffer(&gradient_contribution_buffer, 0, &gradient_contribution_return_buffer, 0, linear_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = gradient_contribution_return_buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        if let Some(Ok(())) = receiver.receive().await {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            gradient_contribution_return_buffer.unmap();
            na::DVector::<F>::from_iterator(result.len(), result.into_iter().map(|f| F::from(f).unwrap()))
        } else {
            panic!("Failed to map return buffer memory");
        }
    }
}

/// Gpu-offloaded distance bounds gradient calculation and parallel otherwise
pub struct Gpu<F: Float> {
    bounds: Bounds<F>,
    gpu_linear_squared_bounds: na::Matrix2xX<f32>,
    gpu_handles: Option<GpuHandles>,
    stage: Stage,
}

impl<F: Float> Gpu<F> {
    /// Transform distance bounds for transfer to gpu
    fn into_matrix(bounds: &[DistanceBound<F>]) -> na::Matrix2xX<f32> {
        let n = bounds.len();
        na::Matrix2xX::<f32>::from_iterator(
            n,
            bounds.iter()
                .flat_map(|bound| {
                    let (lower, upper) = bound.square_bounds;
                    [upper.to_f32().unwrap(), lower.to_f32().unwrap()].into_iter() // access order in shader
                })
        )
    }

    /// Construct from bounds
    pub fn new(bounds: Bounds<F>) -> Gpu<F> {
        let gpu_linear_squared_bounds = Self::into_matrix(&bounds.distances);
        let gpu_handles = GpuHandles::new();
        Gpu {bounds, gpu_linear_squared_bounds, gpu_handles, stage: Stage::FixChirals}
    }

    /// Run compute shader on gpu to calculate distance gradient contributions
    async fn distance_gradient(&self, positions: &na::DVector<F>) -> Option<na::DVector<F>> {
        let handles = self.gpu_handles.as_ref()?;
        let contributions = handles.distance_gradient_contributions(positions, &self.gpu_linear_squared_bounds).await;

        // Collect contributions into gradient serially
        let num_pairs = contributions.len() / DIMENSIONS;
        let contributions = contributions.reshape_generic(na::Const::<DIMENSIONS>, na::Dyn(num_pairs));

        let mut gradient = na::DVector::<F>::zeros(positions.nrows());
        for (contribution, bound) in contributions.column_iter().zip(self.bounds.distances.iter()) {
            // TODO try perf with and without this
            if contribution.iter().all(|v| *v == F::zero()) {
                continue;
            }

            {
                let mut part = four_mut(&mut gradient, bound.indices.0);
                part += contribution;
            }
            {
                let mut part = four_mut(&mut gradient, bound.indices.1);
                part -= contribution;
            }
        }

        Some(gradient)
    }

    /// Wraps parallel computation of chiral gradient
    async fn chiral_gradient(&self, positions: &na::DVector<F>) -> na::DVector<F> {
        Parallel::chiral_gradient(&self.bounds.chirals, positions)
    }

    /// Calculate gradients asynchronously, offloading distance gradients to gpu if possible
    async fn async_gradients(&self, positions: &na::DVector<F>) -> na::DVector<F> {
        let distance_fut = self.distance_gradient(positions); // on GPU
        let chiral_fut = self.chiral_gradient(positions); // parallel on CPU
        let (maybe_distance_grad, chiral_grad) = futures::join!(distance_fut, chiral_fut);
        // Fallback onto parallel impl if gpu doesn't work
        let distance_grad = maybe_distance_grad.unwrap_or_else(|| Parallel::distance_gradient(&self.bounds.distances, positions));
        let grad = distance_grad + chiral_grad;
        Parallel::fourth_dimension_gradient(positions, grad, &self.stage)
    }
}

impl<F: Float> RefinementErrorFunction<F> for Gpu<F> {
    fn error(&self, positions: &na::DVector<F>) -> F {
        Parallel::distance_error(&self.bounds.distances, positions)
            + Parallel::chiral_error(&self.bounds.chirals, positions)
            + Parallel::fourth_dimension_error(positions, &self.stage)
    }

    fn gradient(&self, positions: &na::DVector<F>) -> na::DVector<F> {
        pollster::block_on(self.async_gradients(positions))
    }

    fn set_stage(&mut self, stage: Stage) {
        self.stage = stage;
    }
}

#[cfg(test)]
mod tests {
    use crate::dg::{DistanceMatrix, MetricMatrix, MetrizationPartiality};
    use crate::dg::refinement::{Serial, Stage};
    use crate::dg::refinement::gpu::*;

    #[test]
    fn gpu_gradient() {
        let shape = &crate::shapes::SQUAREANTIPRISM;
        let bounds = crate::dg::modeling::solitary_shape::shape_into_bounds(shape);
        let distances = DistanceMatrix::try_from_distance_bounds(bounds.clone(), MetrizationPartiality::Complete).expect("Successful metrization");
        let coords = MetricMatrix::from(distances).embed().scale(0.7);
        let n = coords.len();
        let linear_coords = coords.reshape_generic(na::Dyn(n), na::Const::<1>);
        let refinement_bounds = Bounds::new(bounds, Vec::new());

        let serial_errf = Serial {bounds: refinement_bounds.clone(), stage: Stage::FixChirals};
        let serial_grad = serial_errf.gradient(&linear_coords);

        let gpu_errf = Gpu::new(refinement_bounds);
        let gpu_grad = gpu_errf.gradient(&linear_coords);

        approx::assert_relative_eq!(serial_grad, gpu_grad, epsilon=1e-4);
    }
}
