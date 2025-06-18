const codeContent = [
    '#!/usr/bin/env python3',
    '"""',
    'Decomposition algorithm with real-time updates.',
    'Implements gradient descent with momentum for optimization.',
    '"""',
    '',
    'import numpy as np',
    'import torch',
    'from typing import Dict, List, Tuple, Optional',
    'from dataclasses import dataclass',
    '',
    '@dataclass',
    'class DecompositionConfig:',
    '    learning_rate: float = 0.001',
    '    momentum: float = 0.9',
    '    epsilon: float = 1e-8',
    '    max_iterations: int = 1000',
    '',
    'def decompose_matrix(matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:',
    '    """',
    '    Perform matrix decomposition using SVD.',
    '    ',
    '    Args:',
    '        matrix: Input matrix to decompose',
    '        rank: Target rank for decomposition',
    '    ',
    '    Returns:',
    '        Tuple of decomposed matrices (U, V)',
    '    """',
    '    U, s, Vt = np.linalg.svd(matrix, full_matrices=False)',
    '    U_reduced = U[:, :rank]',
    '    s_reduced = s[:rank]',
    '    Vt_reduced = Vt[:rank, :]',
    '    ',
    '    return U_reduced @ np.diag(s_reduced), Vt_reduced',
    '',
    'class GradientOptimizer:',
    '    def __init__(self, config: DecompositionConfig):',
    '        self.config = config',
    '        self.momentum_buffer = None',
    '        self.iteration = 0',
    '    ',
    '    def step(self, gradient: np.ndarray) -> np.ndarray:',
    '        if self.momentum_buffer is None:',
    '            self.momentum_buffer = np.zeros_like(gradient)',
    '        ',
    '        self.momentum_buffer = (self.config.momentum * self.momentum_buffer + ',
    '                               gradient)',
    '        ',
    '        update = self.config.learning_rate * self.momentum_buffer',
    '        self.iteration += 1',
    '        ',
    '        return update',
    '',
    'def compute_loss(prediction: np.ndarray, target: np.ndarray) -> float:',
    '    """Compute reconstruction loss."""',
    '    return np.mean((prediction - target) ** 2)',
    '',
    'def train_decomposition(data: np.ndarray, epochs: int = 100):',
    '    """Main training loop for decomposition."""',
    '    config = DecompositionConfig()',
    '    optimizer = GradientOptimizer(config)',
    '    ',
    '    for epoch in range(epochs):',
    '        # Forward pass',
    '        U, V = decompose_matrix(data, rank=10)',
    '        reconstruction = U @ V',
    '        ',
    '        # Compute loss and gradients',
    '        loss = compute_loss(reconstruction, data)',
    '        gradient = 2 * (reconstruction - data) / data.size',
    '        ',
    '        # Update parameters',
    '        update = optimizer.step(gradient)',
    '        ',
    '        if epoch % 10 == 0:',
    '            print(f"Epoch {epoch}, Loss: {loss:.6f}")',
    '',
    'if __name__ == "__main__":',
    '    # Generate test data',
    '    np.random.seed(42)',
    '    test_matrix = np.random.randn(100, 50)',
    '    ',
    '    # Run decomposition',
    '    train_decomposition(test_matrix, epochs=200)'
];

const docData = {
    doc1: [
        'decompose_matrix(matrix, rank)',
        '  Performs SVD-based matrix decomposition',
        '  Parameters:',
        '    matrix: numpy.ndarray',
        '    rank: int',
        '  Returns: Tuple[ndarray, ndarray]',
        '',
        'GradientOptimizer.step(gradient)',
        '  Updates parameters using momentum',
        '  Parameters:',
        '    gradient: numpy.ndarray',
        '  Returns: numpy.ndarray',
        '',
        'compute_loss(prediction, target)',
        '  Calculates MSE reconstruction loss',
        '  Parameters:',
        '    prediction: numpy.ndarray',
        '    target: numpy.ndarray',
        '  Returns: float',
        '',
        'train_decomposition(data, epochs)',
        '  Main training loop implementation',
        '  Parameters:',
        '    data: numpy.ndarray',
        '    epochs: int = 100',
        '  Returns: None'
    ],
    doc2: [
        'Matrix Decomposition:',
        '- Uses truncated SVD for efficiency',
        '- Rank should be << min(m,n)',
        '- Memory usage scales as O(rank*m)',
        '',
        'Time Complexity: O(min(m²n, mn²))',
        'Space Complexity: O(mn)',
        '',
        'Implementation Details:',
        '- Truncates to specified rank',
        '- Returns U_reduced and V_reduced',
        '- Maintains numerical stability',
        '',
        'Optimization Strategy:',
        '- Momentum helps escape local minima',
        '- Learning rate scheduling recommended',
        '- Gradient clipping may be beneficial',
        '',
        'Mathematical Formula:',
        'm_t = β₁m_{t-1} + ∇f(θ)',
        'θ_{t+1} = θ_t - α * m_t',
        '',
        'Where:',
        '- β₁ is momentum coefficient',
        '- α is learning rate',
        '- ∇f(θ) is gradient'
    ],
    doc3: [
        'Basic Usage:',
        '',
        'import numpy as np',
        'from decomposition import train_decomposition',
        '',
        '# Create test data',
        'data = np.random.randn(1000, 100)',
        'train_decomposition(data, epochs=50)',
        '',
        'Advanced Configuration:',
        '',
        'config = DecompositionConfig(',
        '    learning_rate=0.01,',
        '    momentum=0.95,',
        '    max_iterations=2000',
        ')',
        '',
        'Custom Optimization:',
        '',
        'optimizer = GradientOptimizer(config)',
        'for i in range(100):',
        '    grad = compute_gradient()',
        '    update = optimizer.step(grad)',
        '    apply_update(update)',
        '',
        'Batch Processing:',
        '',
        'for batch in data_loader:',
        '    U, V = decompose_matrix(batch, rank=20)',
        '    loss = compute_loss(U @ V, batch)',
        '    print(f"Batch loss: {loss}")'
    ]
};

let animationRunning = true;
let animationSpeed = 500; // 2 changes per second
let iterationCount = 1;
let timeoutId;

const newCodeLines = [
    'from sklearn.decomposition import PCA, TruncatedSVD',
    'from sklearn.preprocessing import StandardScaler',
    'import matplotlib.pyplot as plt',
    'import seaborn as sns',
    'import logging',
    'from concurrent.futures import ThreadPoolExecutor',
    'from functools import lru_cache',
    'import warnings',
    '',
    'logger = logging.getLogger(__name__)',
    'warnings.filterwarnings("ignore", category=FutureWarning)',
    '',
    'class ValidationError(Exception):',
    '    """Custom exception for validation errors."""',
    '    pass',
    '',
    'def validate_input(matrix: np.ndarray) -> bool:',
    '    """',
    '    Comprehensive input validation for matrices.',
    '    ',
    '    Checks:',
    '    - Matrix dimensions',
    '    - NaN/Inf values',
    '    - Data type compatibility',
    '    """',
    '    if matrix.ndim != 2:',
    '        raise ValidationError(f"Expected 2D array, got {matrix.ndim}D")',
    '    ',
    '    if matrix.size == 0:',
    '        raise ValidationError("Empty matrix provided")',
    '    ',
    '    if np.any(np.isnan(matrix)) or np.any(np.isinf(matrix)):',
    '        raise ValidationError("Matrix contains NaN or Inf values")',
    '    ',
    '    return True',
    '',
    '@lru_cache(maxsize=128)',
    'def compute_svd_threshold(matrix_shape: Tuple[int, int], rank: int) -> float:',
    '    """Calculate optimal threshold for SVD truncation."""',
    '    m, n = matrix_shape',
    '    return rank * np.sqrt(2 * (m + n) + 1)',
    '',
    'class AdaptiveLearningRate:',
    '    """',
    '    Implements various learning rate scheduling strategies.',
    '    ',
    '    Supports:',
    '    - Exponential decay',
    '    - Step decay',
    '    - Cosine annealing',
    '    - Plateau-based reduction',
    '    """',
    '    ',
    '    def __init__(self, initial_lr: float = 0.001, strategy: str = "exponential"):',
    '        self.initial_lr = initial_lr',
    '        self.current_lr = initial_lr',
    '        self.strategy = strategy',
    '        self.iteration = 0',
    '        self.best_loss = float("inf")',
    '        self.patience_counter = 0',
    '    ',
    '    def step(self, current_loss: Optional[float] = None) -> float:',
    '        """Update learning rate based on strategy."""',
    '        self.iteration += 1',
    '        ',
    '        if self.strategy == "exponential":',
    '            self.current_lr = self.initial_lr * np.exp(-0.1 * self.iteration)',
    '        elif self.strategy == "step":',
    '            self.current_lr = self.initial_lr * (0.1 ** (self.iteration // 30))',
    '        elif self.strategy == "cosine":',
    '            self.current_lr = self.initial_lr * (1 + np.cos(np.pi * self.iteration / 100)) / 2',
    '        elif self.strategy == "plateau" and current_loss is not None:',
    '            if current_loss < self.best_loss:',
    '                self.best_loss = current_loss',
    '                self.patience_counter = 0',
    '            else:',
    '                self.patience_counter += 1',
    '                if self.patience_counter > 10:',
    '                    self.current_lr *= 0.5',
    '                    self.patience_counter = 0',
    '        ',
    '        return self.current_lr',
    '',
    'def parallel_matrix_operations(matrices: List[np.ndarray], operation: callable) -> List[np.ndarray]:',
    '    """Execute matrix operations in parallel."""',
    '    with ThreadPoolExecutor(max_workers=4) as executor:',
    '        results = list(executor.map(operation, matrices))',
    '    return results',
    '',
    'def plot_convergence(losses: List[float], title: str = "Training Convergence"):',
    '    """',
    '    Create publication-quality convergence plots.',
    '    """',
    '    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))',
    '    ',
    '    # Loss over iterations',
    '    ax1.plot(losses, linewidth=2)',
    '    ax1.set_xlabel("Iteration")',
    '    ax1.set_ylabel("Loss")',
    '    ax1.set_title(title)',
    '    ax1.grid(True, alpha=0.3)',
    '    ax1.set_yscale("log")',
    '    ',
    '    # Loss distribution',
    '    ax2.hist(losses, bins=50, alpha=0.7, edgecolor="black")',
    '    ax2.set_xlabel("Loss Value")',
    '    ax2.set_ylabel("Frequency")',
    '    ax2.set_title("Loss Distribution")',
    '    ',
    '    plt.tight_layout()',
    '    plt.show()',
    '',
    'class RobustDecomposition:',
    '    """',
    '    Robust matrix decomposition with multiple backend support.',
    '    """',
    '    ',
    '    def __init__(self, method: str = "svd"):',
    '        self.method = method',
    '        self.history = {"loss": [], "rank": [], "time": []}',
    '    ',
    '    def fit(self, matrix: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:',
    '        """Fit decomposition model."""',
    '        start_time = time.time()',
    '        ',
    '        if self.method == "svd":',
    '            U, V = self._svd_decomposition(matrix, rank)',
    '        elif self.method == "nmf":',
    '            U, V = self._nmf_decomposition(matrix, rank)',
    '        elif self.method == "pca":',
    '            U, V = self._pca_decomposition(matrix, rank)',
    '        else:',
    '            raise ValueError(f"Unknown method: {self.method}")',
    '        ',
    '        elapsed = time.time() - start_time',
    '        self.history["time"].append(elapsed)',
    '        ',
    '        return U, V',
    '',
    '# Add regularization and constraints',
    'def add_l1_regularization(weights: np.ndarray, lambda_l1: float) -> float:',
    '    """L1 regularization term."""',
    '    return lambda_l1 * np.sum(np.abs(weights))',
    '',
    'def add_l2_regularization(weights: np.ndarray, lambda_l2: float) -> float:',
    '    """L2 regularization term."""',
    '    return lambda_l2 * np.sum(weights ** 2)',
    '',
    'def elastic_net_regularization(weights: np.ndarray, lambda_l1: float, lambda_l2: float, alpha: float) -> float:',
    '    """Elastic net combining L1 and L2."""',
    '    l1_term = add_l1_regularization(weights, lambda_l1)',
    '    l2_term = add_l2_regularization(weights, lambda_l2)',
    '    return alpha * l1_term + (1 - alpha) * l2_term'
];

const newDocEntries = {
    doc1: [
        'ValidationError',
        '  Custom exception for matrix validation',
        '  Inherits from: Exception',
        '',
        'compute_svd_threshold(shape, rank)',
        '  Calculates optimal SVD truncation threshold',
        '  Uses: Marchenko-Pastur distribution',
        '  Returns: float threshold value',
        '',
        'AdaptiveLearningRate',
        '  Learning rate scheduler with multiple strategies',
        '  Methods: step(), reset(), get_history()',
        '',
        'RobustDecomposition',
        '  Multi-backend matrix decomposition',
        '  Backends: SVD, NMF, PCA, ICA',
        '  Methods: fit(), transform(), fit_transform()',
        '',
        'parallel_matrix_operations(matrices, op)',
        '  Concurrent matrix processing',
        '  Max workers: 4',
        '  Returns: List[ndarray]',
        '',
        'elastic_net_regularization(weights, l1, l2, alpha)',
        '  Combined L1/L2 regularization',
        '  Alpha: mixing parameter [0,1]'
    ],
    doc2: [
        'Performance Optimization Strategies:',
        '',
        'Memory Management:',
        '- Use np.float32 for large matrices',
        '- Implement chunked processing for > 10GB',
        '- Clear intermediate results with del',
        '- Monitor memory usage with tracemalloc',
        '',
        'Computational Efficiency:',
        '- Leverage BLAS/LAPACK backends',
        '- Use scipy.sparse for sparse matrices',
        '- Enable MKL threading: export MKL_NUM_THREADS=4',
        '- Profile with line_profiler',
        '',
        'Numerical Stability:',
        '- Condition number checking',
        '- Iterative refinement for ill-conditioned problems',
        '- Use np.linalg.lstsq for overdetermined systems',
        '- Implement gradient clipping: torch.nn.utils.clip_grad_norm_',
        '',
        'Distributed Computing:',
        '- Dask for out-of-core computation',
        '- Ray for distributed training',
        '- Horovod for multi-GPU setups',
        '',
        'Algorithm Selection:',
        '- SVD: Best for general decomposition',
        '- Randomized SVD: For approximate solutions',
        '- QR: When orthogonality is critical',
        '- Cholesky: For positive definite matrices'
    ],
    doc3: [
        '# Advanced Usage Patterns',
        '',
        '## Distributed Decomposition',
        'import dask.array as da',
        'from dask.distributed import Client',
        '',
        'client = Client("scheduler:8786")',
        'x_dask = da.from_array(large_matrix, chunks=(1000, 1000))',
        'u, s, v = da.linalg.svd(x_dask)',
        'result = u.compute()',
        '',
        '## GPU Acceleration',
        'import cupy as cp',
        '',
        'gpu_matrix = cp.asarray(matrix)',
        'u_gpu, s_gpu, v_gpu = cp.linalg.svd(gpu_matrix)',
        'result = cp.asnumpy(u_gpu)',
        '',
        '## Incremental SVD',
        'from sklearn.decomposition import IncrementalPCA',
        '',
        'ipca = IncrementalPCA(n_components=10, batch_size=100)',
        'for batch in data_generator():',
        '    ipca.partial_fit(batch)',
        '',
        '## Robust PCA (RPCA)',
        'def robust_pca(M, lambda_=None):',
        '    """Decompose M = L + S (low-rank + sparse)"""',
        '    if lambda_ is None:',
        '        lambda_ = 1 / np.sqrt(max(M.shape))',
        '    # ... implementation ...',
        '',
        '## Cross-validation for rank selection',
        'from sklearn.model_selection import cross_val_score',
        '',
        'ranks = range(5, 50, 5)',
        'scores = []',
        'for rank in ranks:',
        '    decomposer = TruncatedSVD(n_components=rank)',
        '    score = cross_val_score(decomposer, X, cv=5)',
        '    scores.append(score.mean())',
        '',
        '## Streaming updates',
        'class StreamingSVD:',
        '    def __init__(self, rank):',
        '        self.rank = rank',
        '        self.mean = None',
        '        self.components = None',
        '    ',
        '    def partial_fit(self, X_batch):',
        '        # Online mean update',
        '        if self.mean is None:',
        '            self.mean = X_batch.mean(axis=0)',
        '        else:',
        '            self.mean = 0.9 * self.mean + 0.1 * X_batch.mean(axis=0)',
        '        # ... update components ...'
    ]
};

function applySyntaxHighlighting(text) {
    return text
        .replace(/\b(def|class|import|from|if|else|elif|for|while|return|try|except|finally|with|as|in|and|or|not|is|lambda|yield|async|await)\b/g, '<span class="keyword">$1</span>')
        .replace(/"""[\s\S]*?"""/g, '<span class="comment">$&</span>')
        .replace(/#.*$/gm, '<span class="comment">$&</span>')
        .replace(/"([^"]*)"/g, '<span class="string">"$1"</span>')
        .replace(/'([^']*)'/g, '<span class="string">\'$1\'</span>')
        .replace(/\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(/g, '<span class="function">$1</span>(')
        .replace(/\b\d+\.?\d*\b/g, '<span class="number">$&</span>');
}

function renderCode() {
    const container = document.getElementById('codeContent');
    container.innerHTML = '';
    
    codeContent.forEach((line, index) => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'code-line';
        
        const lineNumber = document.createElement('div');
        lineNumber.className = 'line-number';
        lineNumber.textContent = index + 1;
        
        const lineContent = document.createElement('div');
        lineContent.className = 'line-content';
        lineContent.innerHTML = applySyntaxHighlighting(line);
        
        lineDiv.appendChild(lineNumber);
        lineDiv.appendChild(lineContent);
        container.appendChild(lineDiv);
    });
}

function renderDoc(docId, content) {
    const container = document.getElementById(docId);
    container.innerHTML = '';
    
    content.forEach(line => {
        const lineDiv = document.createElement('div');
        lineDiv.className = 'doc-line';
        lineDiv.textContent = line;
        container.appendChild(lineDiv);
    });
}

function scrollToLine(containerId, lineIndex) {
    const container = document.getElementById(containerId);
    const lines = container.querySelectorAll(containerId === 'codeContent' ? '.code-line' : '.doc-line');
    
    if (lines[lineIndex]) {
        const line = lines[lineIndex];
        const containerRect = container.getBoundingClientRect();
        const lineRect = line.getBoundingClientRect();
        
        const scrollTop = container.scrollTop + lineRect.top - containerRect.top - containerRect.height / 2 + lineRect.height / 2;
        
        container.scrollTo({
            top: Math.max(0, scrollTop),
            behavior: 'smooth'
        });
    }
}

function animateChanges() {
    if (!animationRunning) return;

    const changeType = Math.random();
    
    if (changeType < 0.5) {
        // Update documentation
        const docIds = ['doc1', 'doc2', 'doc3'];
        const randomDoc = docIds[Math.floor(Math.random() * docIds.length)];
        const docContent = docData[randomDoc];
        const newEntries = newDocEntries[randomDoc];
        
        if (Math.random() < 0.7) {
            // Add large blocks of documentation (10-20 lines)
            const blockSize = Math.floor(Math.random() * 11) + 10; // 10-20 lines
            const insertIndex = Math.floor(Math.random() * (docContent.length + 1));
            
            for (let i = 0; i < blockSize; i++) {
                const newEntry = newEntries[Math.floor(Math.random() * newEntries.length)];
                docContent.splice(insertIndex + i, 0, newEntry);
            }
            
            renderDoc(randomDoc, docContent);
            
            setTimeout(() => {
                scrollToLine(randomDoc, insertIndex + Math.floor(blockSize / 2));
                const lines = document.getElementById(randomDoc).querySelectorAll('.doc-line');
                for (let i = 0; i < blockSize; i++) {
                    if (lines[insertIndex + i]) {
                        lines[insertIndex + i].classList.add('added');
                        setTimeout(() => {
                            lines[insertIndex + i].classList.remove('added');
                        }, 2000);
                    }
                }
            }, 50);
        } else if (docContent.length > 10) {
            // Remove blocks of documentation (but not too much)
            const maxDeleteSize = Math.min(15, Math.floor(docContent.length * 0.3));
            const deleteSize = Math.min(Math.floor(Math.random() * 6) + 5, maxDeleteSize);
            const deleteIndex = Math.floor(Math.random() * Math.max(1, docContent.length - deleteSize));
            const lines = document.getElementById(randomDoc).querySelectorAll('.doc-line');
            
            scrollToLine(randomDoc, deleteIndex + Math.floor(deleteSize / 2));
            
            for (let i = 0; i < deleteSize; i++) {
                if (lines[deleteIndex + i]) {
                    lines[deleteIndex + i].classList.add('deleted');
                }
            }
            
            setTimeout(() => {
                docContent.splice(deleteIndex, deleteSize);
                renderDoc(randomDoc, docContent);
            }, 1000);
        }
    } else {
        // Update main code
        if (Math.random() < 0.7) {
            // Add large blocks of code (10-20 lines)
            const blockSize = Math.floor(Math.random() * 11) + 10; // 10-20 lines
            const insertIndex = Math.floor(Math.random() * (codeContent.length + 1));
            
            for (let i = 0; i < blockSize; i++) {
                const newLine = newCodeLines[Math.floor(Math.random() * newCodeLines.length)];
                codeContent.splice(insertIndex + i, 0, newLine);
            }
            
            renderCode();
            
            setTimeout(() => {
                scrollToLine('codeContent', insertIndex + Math.floor(blockSize / 2));
                const lines = document.getElementById('codeContent').querySelectorAll('.code-line');
                for (let i = 0; i < blockSize; i++) {
                    if (lines[insertIndex + i]) {
                        lines[insertIndex + i].classList.add('added');
                        setTimeout(() => {
                            lines[insertIndex + i].classList.remove('added');
                        }, 2000);
                    }
                }
            }, 50);
        } else if (codeContent.length > 10) {
            // Sometimes modify, sometimes delete
            if (Math.random() < 0.5) {
                // Delete blocks of lines (but not too much)
                const maxDeleteSize = Math.min(15, Math.floor(codeContent.length * 0.3));
                const deleteSize = Math.min(Math.floor(Math.random() * 6) + 5, maxDeleteSize);
                const deleteIndex = Math.floor(Math.random() * Math.max(1, codeContent.length - deleteSize));
                
                scrollToLine('codeContent', deleteIndex + Math.floor(deleteSize / 2));
                
                const lines = document.getElementById('codeContent').querySelectorAll('.code-line');
                for (let i = 0; i < deleteSize; i++) {
                    if (lines[deleteIndex + i]) {
                        lines[deleteIndex + i].classList.add('deleted');
                    }
                }
                
                setTimeout(() => {
                    codeContent.splice(deleteIndex, deleteSize);
                    renderCode();
                }, 1000);
            } else {
                // Modify block of lines (5-10)
                const modifySize = Math.floor(Math.random() * 6) + 5;
                const modifyIndex = Math.floor(Math.random() * Math.max(1, codeContent.length - modifySize));
                
                scrollToLine('codeContent', modifyIndex + Math.floor(modifySize / 2));
                
                for (let i = 0; i < modifySize && (modifyIndex + i) < codeContent.length; i++) {
                    const originalLine = codeContent[modifyIndex + i];
                    const modifications = [
                        originalLine + '  # Updated',
                        originalLine.replace(/matrix/g, 'data'),
                        originalLine.replace(/rank/g, 'components'),
                        originalLine.replace(/gradient/g, 'grad'),
                        originalLine.replace(/def /g, 'async def '),
                        originalLine.replace(/float/g, 'np.float64'),
                        originalLine.replace(/return/g, 'yield'),
                    ];
                    
                    codeContent[modifyIndex + i] = modifications[Math.floor(Math.random() * modifications.length)];
                }
                
                renderCode();
                
                setTimeout(() => {
                    const lines = document.getElementById('codeContent').querySelectorAll('.code-line');
                    for (let i = 0; i < modifySize; i++) {
                        if (lines[modifyIndex + i]) {
                            lines[modifyIndex + i].classList.add('modified');
                            setTimeout(() => {
                                lines[modifyIndex + i].classList.remove('modified');
                            }, 2000);
                        }
                    }
                }, 50);
            }
        }
    }
    
    // Update iteration counter
    iterationCount++;
    document.getElementById('iterationCount').textContent = iterationCount;
    
    // Schedule next change at fixed interval
    timeoutId = setTimeout(animateChanges, animationSpeed);
}

function toggleAnimation() {
    animationRunning = !animationRunning;
    const btn = document.getElementById('pauseBtn');
    
    if (animationRunning) {
        btn.textContent = 'Pause';
        btn.classList.remove('paused');
        animateChanges();
    } else {
        btn.textContent = 'Resume';
        btn.classList.add('paused');
        clearTimeout(timeoutId);
    }
}

function changeSpeed() {
    // Speed is now fixed at 2 changes per second
    // This button could be repurposed or removed
}

function resetContent() {
    iterationCount = 1;
    document.getElementById('iterationCount').textContent = iterationCount;
    
    // Reset all content
    renderCode();
    Object.keys(docData).forEach(docId => {
        renderDoc(docId, docData[docId]);
    });
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    renderCode();
    Object.keys(docData).forEach(docId => {
        renderDoc(docId, docData[docId]);
    });
    
    setTimeout(animateChanges, 1000);
});
