import logging
import time
from typing import Dict, List, Optional, Union

import mlflow
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator, QiskitRuntimeService, Sampler, Session

logger = logging.getLogger(__name__)

# Cursor Task: IBM Quantum hardware integration with safety limits


class IBMQuantumHardware:
    """
    IBM Quantum hardware integration with safety limits and time guards.

    Provides controlled access to real quantum hardware with automatic
    fallback to simulators and time-based execution limits.
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        channel: str = "ibm_quantum",
        instance: str = "ibm-q/open/main",
        max_execution_minutes: int = 10,
        mlflow_logging: bool = True,
    ):
        """
        Initialize IBM Quantum hardware access.

        Parameters
        ----------
        api_token : str, optional
            IBM Quantum API token. If None, will try to load from saved account.
        channel : str, default="ibm_quantum"
            IBM Quantum channel ("ibm_quantum" or "ibm_cloud").
        instance : str, default="ibm-q/open/main"
            IBM Quantum instance in format "hub/group/project".
        max_execution_minutes : int, default=10
            Maximum execution time in minutes before timeout.
        mlflow_logging : bool, default=True
            If True, log hardware usage to MLflow.
        """
        self.api_token = api_token
        self.channel = channel
        self.instance = instance
        self.max_execution_minutes = max_execution_minutes
        self.mlflow_logging = mlflow_logging

        self.service = None
        self.backend = None
        self.session = None
        self._execution_start_time = None

        logger.info(
            f"Initialized IBM Quantum hardware with {max_execution_minutes}min timeout"
        )

    def connect(self) -> bool:
        """
        Connect to IBM Quantum services.

        Returns
        -------
        bool
            True if connection successful, False otherwise.
        """
        try:
            # Initialize service
            if self.api_token:
                self.service = QiskitRuntimeService(
                    channel=self.channel, token=self.api_token, instance=self.instance
                )
            else:
                # Try to load from saved account
                self.service = QiskitRuntimeService(
                    channel=self.channel, instance=self.instance
                )

            logger.info(f"Connected to IBM Quantum: {self.instance}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to IBM Quantum: {e}")
            return False

    def get_available_backends(self) -> List[Dict[str, Union[str, int, bool]]]:
        """
        Get list of available quantum backends.

        Returns
        -------
        List[Dict[str, Union[str, int, bool]]]
            List of backend information dictionaries.
        """
        if not self.service:
            logger.warning("Not connected to IBM Quantum. Call connect() first.")
            return []

        backends = []
        for backend in self.service.backends():
            backend_info = {
                "name": backend.name,
                "status": backend.status().status_msg,
                "pending_jobs": backend.status().pending_jobs,
                "num_qubits": backend.configuration().n_qubits,
                "is_simulator": backend.configuration().simulator,
                "is_operational": backend.status().operational,
            }
            backends.append(backend_info)

        logger.info(f"Found {len(backends)} available backends")
        return backends

    def select_backend(self, backend_name: str) -> bool:
        """
        Select a specific backend for execution.

        Parameters
        ----------
        backend_name : str
            Name of the backend to select.

        Returns
        -------
        bool
            True if backend selected successfully, False otherwise.
        """
        if not self.service:
            logger.error("Not connected to IBM Quantum. Call connect() first.")
            return False

        try:
            self.backend = self.service.get_backend(backend_name)

            # Check if backend is operational
            if not self.backend.status().operational:
                logger.warning(f"Backend {backend_name} is not operational")
                return False

            # Check queue length
            pending_jobs = self.backend.status().pending_jobs
            if pending_jobs > 10:  # Arbitrary threshold
                logger.warning(
                    f"Backend {backend_name} has {pending_jobs} pending jobs"
                )

            logger.info(
                f"Selected backend: {backend_name} ({self.backend.configuration().n_qubits} qubits)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to select backend {backend_name}: {e}")
            return False

    def start_execution_timer(self) -> None:
        """Start the execution timer for time guard."""
        self._execution_start_time = time.time()
        logger.info(
            f"Started execution timer (max {self.max_execution_minutes} minutes)"
        )

    def check_execution_timeout(self) -> bool:
        """
        Check if execution has exceeded the time limit.

        Returns
        -------
        bool
            True if timeout exceeded, False otherwise.
        """
        if self._execution_start_time is None:
            return False

        elapsed_minutes = (time.time() - self._execution_start_time) / 60.0
        if elapsed_minutes > self.max_execution_minutes:
            logger.warning(f"Execution timeout exceeded: {elapsed_minutes:.1f} minutes")
            return True

        return False

    def execute_circuit(
        self,
        circuit: QuantumCircuit,
        shots: int = 1024,
        optimization_level: int = 1,
    ) -> Optional[Dict[str, Union[int, float]]]:
        """
        Execute a quantum circuit on the selected backend.

        Parameters
        ----------
        circuit : QuantumCircuit
            The quantum circuit to execute.
        shots : int, default=1024
            Number of shots for execution.
        optimization_level : int, default=1
            Transpilation optimization level (0-3).

        Returns
        -------
        Optional[Dict[str, Union[int, float]]]
            Execution results or None if failed.
        """
        if not self.backend:
            logger.error("No backend selected. Call select_backend() first.")
            return None

        if self.check_execution_timeout():
            logger.error("Execution aborted due to timeout")
            return None

        try:
            # Start execution timer
            self.start_execution_timer()

            # Create session with the backend
            with Session(service=self.service, backend=self.backend) as session:
                # Create sampler
                sampler = Sampler(session=session)

                # Submit job
                job = sampler.run([circuit], shots=shots)
                job_id = job.job_id()

                logger.info(f"Submitted job {job_id} to {self.backend.name}")

                # Monitor job with timeout
                result = self._monitor_job(job)

                if result is None:
                    logger.error("Job execution failed or timed out")
                    return None

                # Process results
                counts = result.quasi_dists[0]

                execution_result = {
                    "job_id": job_id,
                    "backend_name": self.backend.name,
                    "shots": shots,
                    "counts": counts,
                    "execution_time_seconds": time.time() - self._execution_start_time,
                }

                if self.mlflow_logging:
                    mlflow.log_param("hardware_backend", self.backend.name)
                    mlflow.log_param("hardware_job_id", job_id)
                    mlflow.log_param("hardware_shots", shots)
                    mlflow.log_metric(
                        "hardware_execution_time_seconds",
                        execution_result["execution_time_seconds"],
                    )
                    mlflow.log_dict(dict(counts), "hardware_counts.json")

                logger.info(f"Job {job_id} completed successfully")
                return execution_result

        except Exception as e:
            logger.error(f"Circuit execution failed: {e}")
            return None

    def _transpile_circuit(
        self, circuit: QuantumCircuit, optimization_level: int
    ) -> QuantumCircuit:
        """
        Transpile circuit for the target backend.

        Parameters
        ----------
        circuit : QuantumCircuit
            Original circuit.
        optimization_level : int
            Optimization level for transpilation.

        Returns
        -------
        QuantumCircuit
            Transpiled circuit.
        """
        from qiskit import transpile

        try:
            transpiled = transpile(
                circuit,
                backend=self.backend,
                optimization_level=optimization_level,
            )

            logger.info(
                f"Transpiled circuit: {len(circuit)} -> {len(transpiled)} gates"
            )
            return transpiled

        except Exception as e:
            logger.error(f"Circuit transpilation failed: {e}")
            raise

    def _monitor_job(self, job) -> Optional[object]:
        """
        Monitor job execution with timeout.

        Parameters
        ----------
        job : RuntimeJob
            The job to monitor.

        Returns
        -------
        Optional[object]
            Job result or None if failed/timed out.
        """
        try:
            # Poll job status with timeout
            timeout_seconds = self.max_execution_minutes * 60
            start_time = time.time()

            while time.time() - start_time < timeout_seconds:
                job_status = job.status()

                if job_status.name == "COMPLETED":
                    return job.result()
                elif job_status.name == "FAILED":
                    logger.error(f"Job failed with error: {job.error_message()}")
                    return None
                elif job_status.name == "CANCELLED":
                    logger.warning("Job was cancelled")
                    return None

                # Check our own timeout
                if self.check_execution_timeout():
                    logger.warning("Cancelling job due to timeout")
                    job.cancel()
                    return None

                # Wait before next poll
                time.sleep(10)  # Poll every 10 seconds

            # Timeout reached
            logger.warning("Job monitoring timeout reached")
            job.cancel()
            return None

        except Exception as e:
            logger.error(f"Job monitoring failed: {e}")
            return None

    def get_fallback_simulator(self) -> AerSimulator:
        """
        Get a fallback simulator for when hardware is unavailable.

        Returns
        -------
        AerSimulator
            Aer simulator instance.
        """
        return AerSimulator(method="automatic")

    def disconnect(self) -> None:
        """Disconnect from IBM Quantum services."""
        if self.service:
            # Close any active session
            if self.session:
                self.session.close()
            # Clear references
            self.service = None
            self.backend = None
            self.session = None
            logger.info("Disconnected from IBM Quantum")


def create_hardware_solver(
    config: Dict[str, Union[str, int, bool]],
    api_token: Optional[str] = None,
) -> Optional[IBMQuantumHardware]:
    """
    Create a hardware solver instance from configuration.

    Parameters
    ----------
    config : Dict[str, Union[str, int, bool]]
        Hardware configuration dictionary.
    api_token : str, optional
        IBM Quantum API token.

    Returns
    -------
    Optional[IBMQuantumHardware]
        Hardware solver instance or None if creation failed.
    """
    if not config.get("enabled", False):
        logger.info("Hardware execution disabled in config")
        return None

    try:
        solver = IBMQuantumHardware(
            api_token=api_token,
            channel=config.get("channel", "ibm_quantum"),
            instance=config.get("instance", "ibm-q/open/main"),
            max_execution_minutes=config.get("max_minutes", 10),
            mlflow_logging=config.get("mlflow_logging", True),
        )

        # Connect and select backend
        if solver.connect():
            backend_name = config.get("backend_name", "ibmq_qasm_simulator")
            if solver.select_backend(backend_name):
                return solver
            else:
                logger.warning(
                    f"Failed to select backend {backend_name}, using fallback simulator"
                )
                return None
        else:
            logger.warning("Failed to connect to IBM Quantum, using fallback simulator")
            return None

    except Exception as e:
        logger.error(f"Failed to create hardware solver: {e}")
        return None
