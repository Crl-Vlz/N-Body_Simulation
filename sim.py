import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import matplotlib.pyplot as plt

class Particle:
    def __init__(self, num_particles, mean = 0, std_dev = 1):
        self.position = (np.random.normal(mean, std_dev, (num_particles, 3)).astype(np.float32) - 0.5) * 1_000  # Use smaller values
        self.velocity = np.zeros((num_particles, 3), dtype=np.float32)
        self.mass = np.random.rand(num_particles).astype(np.float32) * 10000  # Ensure masses are non-zero and not too large

kernel_code = """
__global__ void calculateForces(float3 *positions, float3 *forces, float *masses, int num_particles) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 myPosition = positions[i];
    float3 force = make_float3(0, 0, 0);

    for (int j = 0; j < num_particles; ++j) {
        if (i != j) {
            float3 otherPosition = positions[j];
            float3 diff = make_float3(otherPosition.x - myPosition.x, otherPosition.y - myPosition.y, otherPosition.z - myPosition.z);
            float distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + 1e-9;  // Avoid divide by zero
            if (distSqr > 1e-9) {  // Check to avoid division by zero
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                force.x += masses[j] * invDist3 * diff.x;
                force.y += masses[j] * invDist3 * diff.y;
                force.z += masses[j] * invDist3 * diff.z;
            }
        }
    }

    forces[i] = force;
}

__global__ void integrate(float3 *positions, float3 *velocities, float3 *forces, float *masses, int num_particles, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_particles) return;

    float3 velocity = velocities[i];
    float3 position = positions[i];
    float3 force = forces[i];
    float mass = masses[i];
    
    float3 acceleration = make_float3(force.x / mass, force.y / mass, force.z / mass);

    velocity.x += acceleration.x * dt;
    velocity.y += acceleration.y * dt;
    velocity.z += acceleration.z * dt;

    position.x += velocity.x * dt;
    position.y += velocity.y * dt;
    position.z += velocity.z * dt;

    velocities[i] = velocity;
    positions[i] = position;
}
"""

def run_simulation(particles, num_particles, num_iterations, dt):
    mod = SourceModule(kernel_code)
    calculate_forces = mod.get_function("calculateForces")
    integrate = mod.get_function("integrate")

    d_positions = cuda.mem_alloc(particles.position.nbytes)
    d_velocities = cuda.mem_alloc(particles.velocity.nbytes)
    d_forces = cuda.mem_alloc(particles.position.nbytes)
    d_masses = cuda.mem_alloc(particles.mass.nbytes)

    cuda.memcpy_htod(d_positions, particles.position)
    cuda.memcpy_htod(d_velocities, particles.velocity)
    cuda.memcpy_htod(d_masses, particles.mass)

    block_size = 256
    grid_size = (num_particles + block_size - 1) // block_size

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(particles.position[:, 0], particles.position[:, 1], particles.position[:, 2], c=np.arange(num_particles), cmap='viridis')
    
    positions_history = [particles.position.copy()]

    for iteration in range(num_iterations):
        calculate_forces(d_positions, d_forces, d_masses, np.int32(num_particles), block=(block_size, 1, 1), grid=(grid_size, 1))
        integrate(d_positions, d_velocities, d_forces, d_masses, np.int32(num_particles), np.float32(dt), block=(block_size, 1, 1), grid=(grid_size, 1))

        cuda.memcpy_dtoh(particles.position, d_positions)
        cuda.memcpy_dtoh(particles.velocity, d_velocities)
        positions_history.append(particles.position.copy())
        
        # Update scatter plot
        sc._offsets3d = (particles.position[:, 0], particles.position[:, 1], particles.position[:, 2])
        sc.set_array(np.linspace(0, 1, num_particles) * (iteration + 1) % 1)
        
        ax.set_xlim([-1e6*.5, 1e6*.5])
        ax.set_ylim([-1e6*.5, 1e6*.5])
        ax.set_zlim([-1e6*.5, 1e6*.5])
        plt.draw()
        plt.pause(0.00001)

        if iteration % 10 == 0:
            print(f"Iteration {iteration}: First 5 positions: {particles.position[:5]}")

    plt.ioff()
    plt.show()

    # Plot trajectories after simulation
    fig_traj = plt.figure()
    ax_traj = fig_traj.add_subplot(111, projection='3d')
    for i in range(num_particles):
        trajectory = np.array([pos[i] for pos in positions_history])
        ax_traj.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2])
    plt.show()

if __name__ == "__main__":
    num_particles = 100
    num_iterations = 1000
    dt = 1000

    particles = Particle(num_particles, std_dev=100)
    run_simulation(particles, num_particles, num_iterations, dt)
