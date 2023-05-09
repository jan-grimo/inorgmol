// @group/@binding map input/output variables from NxM numeric space to a local
// variable name
@group(0)
@binding(0)
var<storage, read_write> v_indices: array<u32>; // both input and output

// The Collatz Conjecture states that for any integer n:
// If n is even, n = n/2
// If n is odd, n = 3n+1
// And repeat this process for each new n, you will always eventually reach 1.
// This function returns how many times this recurrence needs to be applied to
// reach 1.
fn collatz_iterations(n_base: u32) -> u32 {
    var n: u32 = n_base;
    var i: u32 = 0u;

    loop {
        if (n <= 1u) {
            break;
        }

        if (n % 2u == 0u) {
            n = n / 2u;
        } else {
            // Overflow? (i.e. 3*n + 1 > 0xffffffffu?)
            if (n >= 1431655765u) {   // 0x55555555u
                return 4294967295u;   // 0xffffffffu
            }

            n = 3u * n + 1u;
        }

        i = i + 1u;
    }

    return i;
}

@compute
@workgroup_size(1) // specifies 'width' in x, y = 1, z = 1 workgroup grid for the shader
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    v_indices[global_id.x] = collatz_iterations(v_indices[global_id.x]);
}

// NOTES
// - @compute marks main as a compute shader stage entry point function
//   (could also be @vertex or @fragment)
// - entry point functions always without a return value
// - @workgroup_size(1) specifies x = 1, y = 1 (default), z = 1 (default)
//   workgroup grid for the shader. This specifies the "dimensionality" and
//   size of the problem the shader is to calculate, and each invocation
//   of the entry function then carries an invocation id to specify which
//   gridpoint in this space is to be worked on
