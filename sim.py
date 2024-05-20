import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

class Particle:
    def __init__(self, num_particles):
        self.position = (np.random.rand(num_particles, 3).astype(np.float32) - 0.5) * 2_000_000
        self.velocity = np.zeros((num_particles, 3), dtype=np.float32)
        self.mass = np.ones(num_particles, dtype=np.float32) * 1_000_000

sim_kernel = SourceModule("""
                             __global__ void calculateForces(float3 *positions, float3 *forces, float *masses, int nParticles) {
                                 int tid = blockIdx.x * blockDim.x + threadIdx.x;
                                 if (tid >= numParticles) return;
                                 
                                 float3 myPosition = positions[tid];
                                 float3 force = make_float3(0, 0, 0);
                                 
                                 for (int j = 0; j < numParticles; ++j) {
                                     if (tid != j) {
                                         float3 otherPosition = positions[j];
                                         float3 diff = make_float3(otherPosition.x - myPosition.x, otherPosition.y - myPosition.y, otherPosition.z - myPosition.z);
                                         float distSqr = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z + 1e-9;
                                         float invDist = rsqrtf(distSqr);
                                         float invDist3 = invDist * invDist * invDist;
                                         
                                         force.x += masses[j] * invDist3 * diff.x;
                                         force.y += masses[j] * invDist3 * diff.y;
                                         force.z += masses[j] * invDist3 * diff.z;
                                         
                                     }
                                 }
                                 
                                 forces[tid] = force;
                             }
                             
                             __global__ void integrate(float3 *positions, float3 *velocities, float3 *forces, float *masses, int numParticles, float dt) {
                                 int tid = blockId.x * blockDim.x + threadIdx.x;
                                 if (tid >= numParticles) return;
                                 
                                 float3 velocity = velocities[tid];
                                 float3 position = positions[tid];
                                 float3 force = forces[tid];
                                 float mass = masses[tid];
                                 
                                 float3 acceleration = make_float3(force.x / mass, force.y / mass, force.z / mass);
                                 
                                 velocity.x += acceleration.x * dt;
                                 velocity.y += acceleration.y * dt;
                                 velocity.z += acceleration.z * dt;
                                 
                                 position.x += velocity.x * dt;
                                 position.y += velocity.y * dt;
                                 position.z += velocity.z * dt;
                                 
                                 velocities[tid] =  velocity;
                                 positions[tid] = position;
                                 
                             }
                             
                             """)

def run(particles : Particle, numParticles, iters, dt):
    calculate_forces = sim_kernel.get_function("calculateForces")
    integrate = sim_kernel.get_function("integrate")
    
    dev_positions = cuda.mem_alloc(particles.position.nbytes)
    dev_velocities = cuda.mem_alloc(particles.velocity.nbytes)
    dev_forces = cuda.mem_alloc(particles.position.nbytes)
    dev_masses = cuda.mem_alloc(particles.mass.nbytes)
    
    cuda.memcpy_htod(dev_positions, particles.position)
    cuda.memcpy_htod(dev_velocities, particles.velocity)
    cuda.memcpy_htod(dev_masses, particles.mass)
    
    block_size = 256
    grid_size = (numParticles + block_size - 1) // block_size
    
    for _ in range(iters):
        calculate_forces(dev_positions, dev_forces, dev_masses, np.int32(numParticles), block = (block_size, 1, 1), grid = (grid_size, 1))
        integrate(dev_positions, dev_velocities, dev_forces, dev_masses, np.int32(numParticles), np.float32(dt), block = (block_size, 1, 1), grid = (grid_size, 1))
        
    cuda.memcpy_dtoh(particles.position, dev_positions)
    cuda.memcpy_dtoh(particles.velocity, dev_velocities)
    
if __name__ == "__main__":
    num_particles = 1000
    num_iterations = 1000
    dt = 0.01

    particles = Particle(num_particles)
    run(particles, num_particles, num_iterations, dt)
    
    for i in range(num_particles):
        print(f"Particle {i + 1} has values:\n\tPosition: ({particles.position[i]})\n\tVelocity: <{particles.velocity[{particles.position[i]}]}>")