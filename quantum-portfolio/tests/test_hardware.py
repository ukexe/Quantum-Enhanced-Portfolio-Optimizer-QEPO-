import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from qiskit import QuantumCircuit

from qepo import hardware


class TestIBMQuantumHardware:
    """Test IBM Quantum hardware integration."""

    def test_hardware_solver_creation_disabled(self):
        """Test creating hardware solver when disabled."""
        config = {"enabled": False}
        solver = hardware.create_hardware_solver(config)
        assert solver is None

    def test_hardware_solver_creation_enabled_no_token(self):
        """Test creating hardware solver when enabled but no token."""
        config = {"enabled": True, "backend_name": "ibmq_qasm_simulator"}
        solver = hardware.create_hardware_solver(config)
        # Should return None due to connection failure (no token)
        assert solver is None

    def test_hardware_solver_initialization(self):
        """Test hardware solver initialization."""
        solver = hardware.IBMQuantumHardware(
            api_token="test_token", max_execution_minutes=5, mlflow_logging=False
        )

        assert solver.api_token == "test_token"
        assert solver.max_execution_minutes == 5
        assert solver.mlflow_logging is False
        assert solver.service is None
        assert solver.backend is None

    def test_execution_timer(self):
        """Test execution timer functionality."""
        solver = hardware.IBMQuantumHardware(mlflow_logging=False)

        # Test timer start
        solver.start_execution_timer()
        assert solver._execution_start_time is not None

        # Test timeout check (should not timeout immediately)
        assert not solver.check_execution_timeout()

    def test_fallback_simulator(self):
        """Test fallback simulator creation."""
        solver = hardware.IBMQuantumHardware(mlflow_logging=False)
        simulator = solver.get_fallback_simulator()

        assert simulator is not None
        assert hasattr(simulator, "run")

    def test_circuit_transpilation(self):
        """Test circuit transpilation with Aer simulator."""
        solver = hardware.IBMQuantumHardware(mlflow_logging=False)

        # Create a simple circuit
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)

        # Use Aer simulator as backend for testing
        from qiskit_aer import AerSimulator

        solver.backend = AerSimulator()

        # Test transpilation
        transpiled = solver._transpile_circuit(circuit, optimization_level=1)
        assert transpiled is not None
        assert len(transpiled) > 0

    def test_disconnect(self):
        """Test disconnection."""
        solver = hardware.IBMQuantumHardware(mlflow_logging=False)
        solver.service = "mock_service"
        solver.backend = "mock_backend"

        solver.disconnect()

        assert solver.service is None
        assert solver.backend is None
        assert solver.session is None


if __name__ == "__main__":
    pytest.main([__file__])
