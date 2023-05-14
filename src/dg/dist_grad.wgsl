// Input: four dimensional positions of all particles
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;

// Input: for each i-j pair linearized squared bounds: x upper bound, y lower bound
@group(0) @binding(1) var<storage, read> linear_squared_bounds: array<vec2<f32>>;

// Output: Gradient contributions of each i-j pair
@group(0) @binding(2) var<storage, write> gradient_contributions: array<vec4<f32>>; 


// Distance error function gradient contribution of one i < j pair
fn gradient(i: u32, j: u32, linear_index: u32) -> vec4<f32> {
  var diff = positions[i] - positions[j];
  var square_distance = dot(diff, diff);

  var bounds = linear_squared_bounds[linear_index];

  var upper_squared = bounds.x;
  var upper_term = square_distance / upper_squared - 1.0;
  if upper_term > 0.0 {
    return (4.0 * upper_term / upper_squared) * diff;
  }

  var lower_squared = bounds.y;
  var quotient = lower_squared + square_distance;
  let lower_term = 2.0 * lower_squared / quotient - 1.0;
  if lower_term > 0.0 {
    return (-8.0 * lower_squared * lower_term / (quotient * quotient)) * diff;
  }

  return vec4<f32>();
}

// Transform upper triangle two-index to linear single index
fn upper_triangle_linear_index(i: u32, j: u32, n: u32) -> u32 {
  return i * n - i * (i + u32(1)) / u32(2) + j - i - u32(1);
}

@compute
@workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if global_id.x < global_id.y {
    var n = arrayLength(&linear_squared_bounds) / u32(4);
    var linear_index = upper_triangle_linear_index(global_id.x, global_id.y, n);
    gradient_contributions[linear_index] = gradient(global_id.x, global_id.y, linear_index);
  }
}
