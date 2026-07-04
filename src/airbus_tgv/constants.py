"""Physical and numerical constants for the Convecting Taylor-Green Vortex benchmark.

Sources: Airbus 2026 Global Quantum + AI Challenge problem statement.
"""
from dataclasses import dataclass
import math

OUTPUT_DISCLAIMER = "Experimental research output – not a validated engineering deliverable."


@dataclass(frozen=True)
class TGVParams:
    """Parameters for the 2D Convecting Taylor-Green Vortex."""
    L: float = 2.0 * math.pi      # Domain length (and characteristic length scale)
    V0: float = 1.0               # Vortex velocity
    Uc: float = 1.0               # Convection x-velocity
    Vc: float = 0.0               # Convection y-velocity
    rho: float = 1.0              # Density
    p0: float = 0.0               # Background pressure
    Re: float = 100.0             # Reynolds number

    @property
    def nu(self) -> float:
        """Kinematic viscosity: nu = V0 * L / Re."""
        return self.V0 * self.L / self.Re

    @property
    def wave_number(self) -> float:
        """Fundamental periodic wave number for a domain of length L."""
        return 2.0 * math.pi / self.L

    @property
    def decay_rate(self) -> float:
        """Velocity decay coefficient for the periodic TGV mode."""
        return 2.0 * self.nu * (self.wave_number ** 2)
