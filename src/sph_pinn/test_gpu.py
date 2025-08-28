import jax
import sys

print(f"JAX running on: {jax.default_backend()}")
if jax.default_backend() != 'gpu':
    print("WARNING: JAX is not using the GPU. Check your JAX installation and CUDA setup.", file=sys.stderr)
else:
    print("✅ JAX is using the GPU.")

# You can also list all available devices
print("Available devices:")
for device in jax.devices():
    print(f"- {device}")