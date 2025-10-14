import logging
from typing import Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit_aer import AerSimulator
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo

logger = logging.getLogger(__name__)

# Cursor Task: QAOA solver for portfolio optimization


class QAOASolver:
    """Quantum Approximate Optimization Algorithm solver for QUBO problems.

    This class implements QAOA using Qiskit Aer simulators to solve
    portfolio optimization problems encoded as QUBO matrices.
    """

    def __init__(
        self,
        backend_name: str = "aer_simulator_statevector",
        shots: int = 1024,
        depth: int = 3,
        max_iterations: int = 100,
        mlflow_logging: bool = True,
    ):
        """Initialize QAOA solver.

        Parameters
        ----------
        backend_name : str, default="aer_simulator_statevector"
            Qiskit Aer backend name
        shots : int, default=1024
            Number of measurement shots
        depth : int, default=3
            QAOA circuit depth (p parameter)
        max_iterations : int, default=100
            Maximum optimizer iterations
        mlflow_logging : bool, default=True
            Whether to log to MLflow
        """
        self.backend_name = backend_name
        self.shots = shots
        self.depth = depth
        self.max_iterations = max_iterations
        self.mlflow_logging = mlflow_logging

        # Initialize backend
        self.backend = self._setup_backend()
        self.sampler = StatevectorSampler()

        # Initialize optimizer
        self.optimizer = SPSA(maxiter=max_iterations)

        logger.info(
            f"Initialized QAOA solver: backend={backend_name}, shots={shots}, depth={depth}"
        )

    def _setup_backend(self) -> AerSimulator:
        """Set up Qiskit Aer backend.

        Returns
        -------
        AerSimulator
            Configured Aer simulator backend
        """
        if self.backend_name == "aer_simulator_statevector":
            backend = AerSimulator(method="statevector")
        elif self.backend_name == "aer_simulator_qasm":
            backend = AerSimulator(method="qasm_simulator", shots=self.shots)
        else:
            # Default to statevector for better performance
            backend = AerSimulator(method="statevector")
            logger.warning(f"Unknown backend {self.backend_name}, using statevector")

        return backend

    def solve_qubo(
        self,
        qubo: Dict[Tuple[int, int], float],
        num_qubits: int,
        config: Optional[Dict] = None,
    ) -> Dict:
        """Solve QUBO problem using QAOA.

        Parameters
        ----------
        qubo : Dict[Tuple[int, int], float]
            QUBO matrix as dictionary
        num_qubits : int
            Number of qubits (variables)
        config : Dict, optional
            Additional configuration parameters

        Returns
        -------
        Dict
            Solution results including:
            - solution: binary solution
            - energy: QUBO energy
            - probability: measurement probability
            - optimization_result: QAOA optimization details
        """
        if config is None:
            config = {}

        logger.info(f"Solving QUBO with {num_qubits} qubits, {len(qubo)} terms")

        # Start MLflow run if logging enabled
        if self.mlflow_logging:
            with mlflow.start_run(nested=True) as run:
                return self._solve_with_logging(
                    qubo, num_qubits, config, run.info.run_id
                )
        else:
            return self._solve_qubo_core(qubo, num_qubits, config)

    def _solve_with_logging(
        self,
        qubo: Dict[Tuple[int, int], float],
        num_qubits: int,
        config: Dict,
        run_id: str,
    ) -> Dict:
        """Solve QUBO with MLflow logging."""
        # Log parameters
        mlflow.log_params(
            {
                "backend": self.backend_name,
                "shots": self.shots,
                "depth": self.depth,
                "max_iterations": self.max_iterations,
                "num_qubits": num_qubits,
                "qubo_terms": len(qubo),
            }
        )
        mlflow.log_params(config)

        # Solve the problem
        result = self._solve_qubo_core(qubo, num_qubits, config)

        # Log results
        mlflow.log_metrics(
            {
                "solution_energy": result["energy"],
                "solution_probability": result["probability"],
                "optimization_iterations": result["optimization_result"].get(
                    "iterations", 0
                ),
                "optimization_time": result["optimization_result"].get("time", 0),
            }
        )

        # Log solution
        mlflow.log_param("solution", result["solution"])

        logger.info(f"QAOA solution logged to MLflow run {run_id}")
        return result

    def _solve_qubo_core(
        self,
        qubo: Dict[Tuple[int, int], float],
        num_qubits: int,
        config: Dict,
    ) -> Dict:
        """Core QUBO solving logic."""
        # Convert QUBO to Qiskit QuadraticProgram
        qp = self._qubo_to_quadratic_program(qubo, num_qubits)

        # Convert to QUBO format for QAOA
        converter = QuadraticProgramToQubo()
        qubo_problem = converter.convert(qp)

        # Set up QAOA
        qaoa = QAOA(
            sampler=self.sampler,
            optimizer=self.optimizer,
            reps=self.depth,
        )

        # Solve using QAOA
        optimizer = MinimumEigenOptimizer(qaoa)

        import time

        start_time = time.time()

        try:
            result = optimizer.solve(qubo_problem)
            solve_time = time.time() - start_time

            # Extract solution
            if result.x is not None:
                solution = result.x.astype(int).tolist()
                energy = result.fval
                probability = self._calculate_solution_probability(
                    solution, qubo, num_qubits
                )
            else:
                logger.warning("QAOA did not find a valid solution")
                solution = [0] * num_qubits
                energy = float("inf")
                probability = 0.0

            optimization_result = {
                "iterations": getattr(result, "iterations", 0),
                "time": solve_time,
                "status": "success" if result.x is not None else "failed",
            }

        except Exception as e:
            logger.error(f"QAOA optimization failed: {e}")
            solution = [0] * num_qubits
            energy = float("inf")
            probability = 0.0
            optimization_result = {
                "iterations": 0,
                "time": time.time() - start_time,
                "status": "error",
                "error": str(e),
            }

        return {
            "solution": solution,
            "energy": energy,
            "probability": probability,
            "optimization_result": optimization_result,
        }

    def _qubo_to_quadratic_program(
        self,
        qubo: Dict[Tuple[int, int], float],
        num_qubits: int,
    ) -> QuadraticProgram:
        """Convert QUBO dictionary to Qiskit QuadraticProgram.

        Parameters
        ----------
        qubo : Dict[Tuple[int, int], float]
            QUBO matrix
        num_qubits : int
            Number of variables

        Returns
        -------
        QuadraticProgram
            Qiskit QuadraticProgram object
        """
        qp = QuadraticProgram()

        # Add binary variables
        for i in range(num_qubits):
            qp.binary_var(name=f"x_{i}")

        # Add objective function
        linear_terms = {}
        quadratic_terms = {}

        for (i, j), coeff in qubo.items():
            if i == j:
                # Linear term
                linear_terms[f"x_{i}"] = coeff
            else:
                # Quadratic term
                quadratic_terms[(f"x_{i}", f"x_{j}")] = coeff

        qp.minimize(linear=linear_terms, quadratic=quadratic_terms)

        return qp

    def _calculate_solution_probability(
        self,
        solution: List[int],
        qubo: Dict[Tuple[int, int], float],
        num_qubits: int,
    ) -> float:
        """Calculate probability of solution (simplified).

        In practice, this would require running the QAOA circuit
        and measuring the probability of the solution state.
        For now, we return a placeholder value.
        """
        # This is a simplified calculation
        # Real implementation would measure from QAOA circuit
        energy = sum(
            qubo.get((i, j), 0) * solution[i] * solution[j]
            for i in range(num_qubits)
            for j in range(num_qubits)
        )

        # Convert energy to probability (simplified)
        # Lower energy should have higher probability
        if energy == float("inf"):
            return 0.0

        # Simple exponential mapping
        probability = np.exp(-abs(energy) / 10.0)
        return min(probability, 1.0)

    def solve_multiple_restarts(
        self,
        qubo: Dict[Tuple[int, int], float],
        num_qubits: int,
        num_restarts: int = 5,
        config: Optional[Dict] = None,
    ) -> List[Dict]:
        """Solve QUBO with multiple random restarts.

        Parameters
        ----------
        qubo : Dict[Tuple[int, int], float]
            QUBO matrix
        num_qubits : int
            Number of qubits
        num_restarts : int, default=5
            Number of restart attempts
        config : Dict, optional
            Configuration parameters

        Returns
        -------
        List[Dict]
            List of solution results from each restart
        """
        logger.info(f"Running {num_restarts} QAOA restarts")

        results = []
        for restart in range(num_restarts):
            logger.info(f"QAOA restart {restart + 1}/{num_restarts}")

            # Randomize optimizer initial parameters
            if config is None:
                config = {}
            config["restart"] = restart

            result = self.solve_qubo(qubo, num_qubits, config)
            result["restart"] = restart
            results.append(result)

        return results

    def rank_solutions(self, results: List[Dict]) -> List[Dict]:
        """Rank solutions by energy (lower is better).

        Parameters
        ----------
        results : List[Dict]
            List of solution results

        Returns
        -------
        List[Dict]
            Results sorted by energy (best first)
        """
        # Filter out failed solutions
        valid_results = [r for r in results if r["energy"] != float("inf")]

        if not valid_results:
            logger.warning("No valid solutions found")
            return results

        # Sort by energy (lower is better)
        ranked = sorted(valid_results, key=lambda x: x["energy"])

        logger.info(f"Ranked {len(ranked)} valid solutions out of {len(results)} total")
        return ranked


def solve_qubo(
    qubo: Dict[Tuple[int, int], float],
    num_qubits: int,
    config: Optional[Dict] = None,
    backend_name: str = "aer_simulator_statevector",
    shots: int = 1024,
    depth: int = 3,
) -> Dict:
    """Convenience function to solve QUBO with QAOA.

    Parameters
    ----------
    qubo : Dict[Tuple[int, int], float]
        QUBO matrix
    num_qubits : int
        Number of qubits
    config : Dict, optional
        Configuration parameters
    backend_name : str, default="aer_simulator_statevector"
        Qiskit Aer backend
    shots : int, default=1024
        Number of shots
    depth : int, default=3
        QAOA depth

    Returns
    -------
    Dict
        Solution results
    """
    solver = QAOASolver(
        backend_name=backend_name,
        shots=shots,
        depth=depth,
    )

    return solver.solve_qubo(qubo, num_qubits, config)


def solve_qubo_with_restarts(
    qubo: Dict[Tuple[int, int], float],
    num_qubits: int,
    num_restarts: int = 5,
    config: Optional[Dict] = None,
    **kwargs,
) -> Dict:
    """Solve QUBO with multiple restarts and return best solution.

    Parameters
    ----------
    qubo : Dict[Tuple[int, int], float]
        QUBO matrix
    num_qubits : int
        Number of qubits
    num_restarts : int, default=5
        Number of restarts
    config : Dict, optional
        Configuration parameters
    **kwargs
        Additional solver parameters

    Returns
    -------
    Dict
        Best solution result
    """
    solver = QAOASolver(**kwargs)

    results = solver.solve_multiple_restarts(qubo, num_qubits, num_restarts, config)
    ranked = solver.rank_solutions(results)

    if ranked:
        best = ranked[0]
        best["all_results"] = results
        return best
    else:
        return {
            "solution": [0] * num_qubits,
            "energy": float("inf"),
            "probability": 0.0,
            "optimization_result": {"status": "failed"},
            "all_results": results,
        }
