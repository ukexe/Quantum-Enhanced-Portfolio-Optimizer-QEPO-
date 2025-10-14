from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from qepo import quantum_qaoa


class TestQAOASolver:
    """Test QAOASolver class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.qubo = {
            (0, 0): -1.0,
            (1, 1): -1.0,
            (0, 1): 0.5,
        }
        self.num_qubits = 2

    @patch("qepo.quantum_qaoa.AerSimulator")
    @patch("qepo.quantum_qaoa.StatevectorSampler")
    @patch("qepo.quantum_qaoa.SPSA")
    def test_solver_initialization(self, mock_spsa, mock_sampler, mock_aer):
        """Test QAOA solver initialization."""
        mock_backend = Mock()
        mock_aer.return_value = mock_backend
        mock_sampler_instance = Mock()
        mock_sampler.return_value = mock_sampler_instance

        solver = quantum_qaoa.QAOASolver(
            backend_name="aer_simulator_statevector",
            shots=512,
            depth=2,
            max_iterations=50,
        )

        assert solver.backend_name == "aer_simulator_statevector"
        assert solver.shots == 512
        assert solver.depth == 2
        assert solver.max_iterations == 50
        assert solver.mlflow_logging is True

    @patch("qepo.quantum_qaoa.AerSimulator")
    @patch("qepo.quantum_qaoa.StatevectorSampler")
    @patch("qepo.quantum_qaoa.SPSA")
    def test_setup_backend_statevector(self, mock_spsa, mock_sampler, mock_aer):
        """Test backend setup for statevector simulator."""
        mock_backend = Mock()
        mock_aer.return_value = mock_backend

        solver = quantum_qaoa.QAOASolver(backend_name="aer_simulator_statevector")
        backend = solver._setup_backend()

        mock_aer.assert_called_with(method="statevector")
        assert backend == mock_backend

    @patch("qepo.quantum_qaoa.AerSimulator")
    @patch("qepo.quantum_qaoa.StatevectorSampler")
    @patch("qepo.quantum_qaoa.SPSA")
    def test_setup_backend_qasm(self, mock_spsa, mock_sampler, mock_aer):
        """Test backend setup for QASM simulator."""
        mock_backend = Mock()
        mock_aer.return_value = mock_backend

        solver = quantum_qaoa.QAOASolver(backend_name="aer_simulator_qasm", shots=1024)
        backend = solver._setup_backend()

        mock_aer.assert_called_with(method="qasm_simulator", shots=1024)
        assert backend == mock_backend

    @patch("qepo.quantum_qaoa.AerSimulator")
    @patch("qepo.quantum_qaoa.StatevectorSampler")
    @patch("qepo.quantum_qaoa.SPSA")
    def test_setup_backend_unknown(self, mock_spsa, mock_sampler, mock_aer):
        """Test backend setup with unknown backend name."""
        mock_backend = Mock()
        mock_aer.return_value = mock_backend

        solver = quantum_qaoa.QAOASolver(backend_name="unknown_backend")
        backend = solver._setup_backend()

        # Should fallback to statevector
        mock_aer.assert_called_with(method="statevector")
        assert backend == mock_backend

    def test_qubo_to_quadratic_program(self):
        """Test QUBO to QuadraticProgram conversion."""
        solver = quantum_qaoa.QAOASolver(mlflow_logging=False)

        qp = solver._qubo_to_quadratic_program(self.qubo, self.num_qubits)

        # Check variables
        assert qp.get_num_vars() == 2
        assert qp.get_num_binary_vars() == 2

        # Check objective function
        obj = qp.objective
        assert obj.sense.name == "MINIMIZE"

    @patch("qepo.quantum_qaoa.AerSimulator")
    @patch("qepo.quantum_qaoa.StatevectorSampler")
    @patch("qepo.quantum_qaoa.SPSA")
    @patch("qepo.quantum_qaoa.QuadraticProgramToQubo")
    @patch("qepo.quantum_qaoa.QAOA")
    @patch("qepo.quantum_qaoa.MinimumEigenOptimizer")
    def test_solve_qubo_core_success(
        self,
        mock_optimizer,
        mock_qaoa,
        mock_converter,
        mock_spsa,
        mock_sampler,
        mock_aer,
    ):
        """Test successful QUBO solving."""
        # Mock the optimization result
        mock_result = Mock()
        mock_result.x = np.array([1, 0])
        mock_result.fval = -1.0
        mock_result.iterations = 10

        mock_optimizer_instance = Mock()
        mock_optimizer_instance.solve.return_value = mock_result
        mock_optimizer.return_value = mock_optimizer_instance

        solver = quantum_qaoa.QAOASolver(mlflow_logging=False)
        result = solver._solve_qubo_core(self.qubo, self.num_qubits, {})

        assert result["solution"] == [1, 0]
        assert result["energy"] == -1.0
        assert result["probability"] > 0
        assert result["optimization_result"]["status"] == "success"
        assert result["optimization_result"]["iterations"] == 10

    @patch("qepo.quantum_qaoa.AerSimulator")
    @patch("qepo.quantum_qaoa.StatevectorSampler")
    @patch("qepo.quantum_qaoa.SPSA")
    @patch("qepo.quantum_qaoa.QuadraticProgramToQubo")
    @patch("qepo.quantum_qaoa.QAOA")
    @patch("qepo.quantum_qaoa.MinimumEigenOptimizer")
    def test_solve_qubo_core_failure(
        self,
        mock_optimizer,
        mock_qaoa,
        mock_converter,
        mock_spsa,
        mock_sampler,
        mock_aer,
    ):
        """Test QUBO solving failure."""
        # Mock failed optimization result
        mock_result = Mock()
        mock_result.x = None
        mock_result.fval = None

        mock_optimizer_instance = Mock()
        mock_optimizer_instance.solve.return_value = mock_result
        mock_optimizer.return_value = mock_optimizer_instance

        solver = quantum_qaoa.QAOASolver(mlflow_logging=False)
        result = solver._solve_qubo_core(self.qubo, self.num_qubits, {})

        assert result["solution"] == [0, 0]
        assert result["energy"] == float("inf")
        assert result["probability"] == 0.0
        assert result["optimization_result"]["status"] == "failed"

    @patch("qepo.quantum_qaoa.AerSimulator")
    @patch("qepo.quantum_qaoa.StatevectorSampler")
    @patch("qepo.quantum_qaoa.SPSA")
    @patch("qepo.quantum_qaoa.QuadraticProgramToQubo")
    @patch("qepo.quantum_qaoa.QAOA")
    @patch("qepo.quantum_qaoa.MinimumEigenOptimizer")
    def test_solve_qubo_core_exception(
        self,
        mock_optimizer,
        mock_qaoa,
        mock_converter,
        mock_spsa,
        mock_sampler,
        mock_aer,
    ):
        """Test QUBO solving with exception."""
        # Mock optimizer to raise exception
        mock_optimizer_instance = Mock()
        mock_optimizer_instance.solve.side_effect = Exception("Optimization failed")
        mock_optimizer.return_value = mock_optimizer_instance

        solver = quantum_qaoa.QAOASolver(mlflow_logging=False)
        result = solver._solve_qubo_core(self.qubo, self.num_qubits, {})

        assert result["solution"] == [0, 0]
        assert result["energy"] == float("inf")
        assert result["probability"] == 0.0
        assert result["optimization_result"]["status"] == "error"
        assert "Optimization failed" in result["optimization_result"]["error"]

    def test_calculate_solution_probability(self):
        """Test solution probability calculation."""
        solver = quantum_qaoa.QAOASolver(mlflow_logging=False)

        # Test with valid solution
        solution = [1, 0]
        probability = solver._calculate_solution_probability(
            solution, self.qubo, self.num_qubits
        )

        assert 0 <= probability <= 1

        # Test with infinite energy
        qubo_inf = {(0, 0): float("inf")}
        probability_inf = solver._calculate_solution_probability([1], qubo_inf, 1)
        assert probability_inf == 0.0

    @patch("qepo.quantum_qaoa.AerSimulator")
    @patch("qepo.quantum_qaoa.StatevectorSampler")
    @patch("qepo.quantum_qaoa.SPSA")
    def test_solve_multiple_restarts(self, mock_spsa, mock_sampler, mock_aer):
        """Test multiple restart solving."""
        solver = quantum_qaoa.QAOASolver(mlflow_logging=False)

        # Mock the solve_qubo method
        mock_results = [
            {
                "solution": [1, 0],
                "energy": -1.0,
                "probability": 0.8,
                "optimization_result": {},
            },
            {
                "solution": [0, 1],
                "energy": -0.5,
                "probability": 0.6,
                "optimization_result": {},
            },
        ]

        with patch.object(solver, "solve_qubo", side_effect=mock_results):
            results = solver.solve_multiple_restarts(
                self.qubo, self.num_qubits, num_restarts=2
            )

        assert len(results) == 2
        assert results[0]["restart"] == 0
        assert results[1]["restart"] == 1

    def test_rank_solutions(self):
        """Test solution ranking."""
        solver = quantum_qaoa.QAOASolver(mlflow_logging=False)

        results = [
            {"solution": [1, 0], "energy": -0.5, "probability": 0.6},
            {"solution": [0, 1], "energy": -1.0, "probability": 0.8},  # Best energy
            {"solution": [1, 1], "energy": float("inf"), "probability": 0.0},  # Invalid
        ]

        ranked = solver.rank_solutions(results)

        assert len(ranked) == 2  # Only valid solutions
        assert ranked[0]["energy"] == -1.0  # Best solution first
        assert ranked[1]["energy"] == -0.5

    def test_rank_solutions_no_valid(self):
        """Test ranking with no valid solutions."""
        solver = quantum_qaoa.QAOASolver(mlflow_logging=False)

        results = [
            {"solution": [1, 0], "energy": float("inf"), "probability": 0.0},
            {"solution": [0, 1], "energy": float("inf"), "probability": 0.0},
        ]

        ranked = solver.rank_solutions(results)

        # Should return original results if no valid solutions
        assert len(ranked) == 2


class TestConvenienceFunctions:
    """Test convenience functions."""

    @patch("qepo.quantum_qaoa.QAOASolver")
    def test_solve_qubo_function(self, mock_solver_class):
        """Test solve_qubo convenience function."""
        mock_solver = Mock()
        mock_solver.solve_qubo.return_value = {"solution": [1, 0], "energy": -1.0}
        mock_solver_class.return_value = mock_solver

        qubo = {(0, 0): -1.0, (1, 1): -1.0}
        result = quantum_qaoa.solve_qubo(qubo, num_qubits=2)

        mock_solver_class.assert_called_once()
        mock_solver.solve_qubo.assert_called_once_with(qubo, 2, None)
        assert result["solution"] == [1, 0]

    @patch("qepo.quantum_qaoa.QAOASolver")
    def test_solve_qubo_with_restarts_function(self, mock_solver_class):
        """Test solve_qubo_with_restarts convenience function."""
        mock_solver = Mock()
        mock_results = [
            {"solution": [1, 0], "energy": -1.0, "probability": 0.8},
            {"solution": [0, 1], "energy": -0.5, "probability": 0.6},
        ]
        mock_solver.solve_multiple_restarts.return_value = mock_results
        mock_solver.rank_solutions.return_value = mock_results
        mock_solver_class.return_value = mock_solver

        qubo = {(0, 0): -1.0, (1, 1): -1.0}
        result = quantum_qaoa.solve_qubo_with_restarts(
            qubo, num_qubits=2, num_restarts=2
        )

        mock_solver_class.assert_called_once()
        mock_solver.solve_multiple_restarts.assert_called_once_with(qubo, 2, 2, None)
        mock_solver.rank_solutions.assert_called_once_with(mock_results)
        assert "all_results" in result

    @patch("qepo.quantum_qaoa.QAOASolver")
    def test_solve_qubo_with_restarts_no_valid_solutions(self, mock_solver_class):
        """Test solve_qubo_with_restarts with no valid solutions."""
        mock_solver = Mock()
        mock_results = [
            {"solution": [0, 0], "energy": float("inf"), "probability": 0.0},
        ]
        mock_solver.solve_multiple_restarts.return_value = mock_results
        mock_solver.rank_solutions.return_value = []  # No valid solutions
        mock_solver_class.return_value = mock_solver

        qubo = {(0, 0): -1.0, (1, 1): -1.0}
        result = quantum_qaoa.solve_qubo_with_restarts(
            qubo, num_qubits=2, num_restarts=1
        )

        assert result["solution"] == [0, 0]
        assert result["energy"] == float("inf")
        assert result["optimization_result"]["status"] == "failed"


class TestMLflowIntegration:
    """Test MLflow integration."""

    @patch("qepo.quantum_qaoa.mlflow.start_run")
    @patch("qepo.quantum_qaoa.AerSimulator")
    @patch("qepo.quantum_qaoa.StatevectorSampler")
    @patch("qepo.quantum_qaoa.SPSA")
    def test_solve_with_logging(self, mock_spsa, mock_sampler, mock_aer, mock_mlflow):
        """Test solving with MLflow logging."""
        # Define test data
        qubo = {
            (0, 0): -1.0,
            (1, 1): -1.0,
            (0, 1): 0.5,
        }
        num_qubits = 2

        # Mock MLflow context manager
        mock_run_info = Mock()
        mock_run_info.run_id = "test-run-id"
        mock_mlflow.return_value.__enter__.return_value = Mock()
        mock_mlflow.return_value.__exit__.return_value = None

        # Mock MLflow logging methods
        with patch("qepo.quantum_qaoa.mlflow.log_params") as mock_log_params, patch(
            "qepo.quantum_qaoa.mlflow.log_metrics"
        ) as mock_log_metrics, patch(
            "qepo.quantum_qaoa.mlflow.log_param"
        ) as mock_log_param:

            solver = quantum_qaoa.QAOASolver(mlflow_logging=True)

            # Mock the core solving method
            with patch.object(solver, "_solve_qubo_core") as mock_solve_core:
                mock_solve_core.return_value = {
                    "solution": [1, 0],
                    "energy": -1.0,
                    "probability": 0.8,
                    "optimization_result": {"iterations": 10, "time": 1.5},
                }

                result = solver.solve_qubo(qubo, num_qubits)

                # Check that MLflow methods were called
                mock_log_params.assert_called()
                mock_log_metrics.assert_called()
                mock_log_param.assert_called()

                assert result["solution"] == [1, 0]
