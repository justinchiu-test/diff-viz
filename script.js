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
    'from sklearn.decomposition import PCA',
    'import matplotlib.pyplot as plt',
    'logger = logging.getLogger(__name__)',
    'def validate_input(matrix: np.ndarray) -> bool:',
    '    return matrix.ndim == 2 and matrix.size > 0',
    'class AdaptiveLearningRate:',
    '    def __init__(self, initial_lr: float = 0.001):',
    '        self.lr = initial_lr',
    'def plot_convergence(losses: List[float]):',
    '    plt.plot(losses)',
    '    plt.xlabel("Iteration")',
    '    plt.ylabel("Loss")',
    '    plt.show()',
    'try:',
    '    result = decompose_matrix(data, rank)',
    'except np.linalg.LinAlgError:',
    '    print("Matrix decomposition failed")',
    '    return None',
    'assert rank > 0, "Rank must be positive"',
    'if data.shape[0] < rank:',
    '    raise ValueError("Rank too large")',
    '# Add regularization term',
    'l2_penalty = 0.001 * np.sum(U**2 + V**2)',
    'total_loss = reconstruction_loss + l2_penalty'
];

const newDocEntries = {
    doc1: [
        'validate_input(matrix)',
        '  Validates input matrix dimensions',
        'plot_convergence(losses)',
        '  Visualizes training progress',
        'AdaptiveLearningRate.update()',
        '  Adjusts learning rate dynamically',
        'PCA.fit_transform(data)',
        '  Alternative decomposition method'
    ],
    doc2: [
        'Validation Best Practices:',
        '- Always check input dimensions',
        '- Verify matrix is not singular',
        '- Handle edge cases gracefully',
        '',
        'Learning Rate Scheduling:',
        '- Start with higher learning rate',
        '- Reduce when loss plateaus',
        '- Monitor validation metrics',
        '',
        'Alternative Methods:',
        '- PCA for exploratory analysis',
        '- Non-negative matrix factorization',
        '- Randomized algorithms for speed'
    ],
    doc3: [
        'Input Validation Example:',
        '',
        'if not validate_input(data):',
        '    raise ValueError("Invalid input")',
        '',
        'Visualization Example:',
        '',
        'losses = []',
        'for epoch in range(epochs):',
        '    loss = train_step()',
        '    losses.append(loss)',
        'plot_convergence(losses)',
        '',
        'Adaptive Learning:',
        '',
        'scheduler = AdaptiveLearningRate(0.01)',
        'for epoch in range(epochs):',
        '    lr = scheduler.get_lr()',
        '    optimizer.lr = lr'
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
    
    if (changeType < 0.6) {
        // Update documentation
        const docIds = ['doc1', 'doc2', 'doc3'];
        const randomDoc = docIds[Math.floor(Math.random() * docIds.length)];
        const docContent = docData[randomDoc];
        const newEntries = newDocEntries[randomDoc];
        
        if (Math.random() < 0.7) {
            // Add new documentation line
            const newEntry = newEntries[Math.floor(Math.random() * newEntries.length)];
            const insertIndex = Math.floor(Math.random() * (docContent.length + 1));
            docContent.splice(insertIndex, 0, newEntry);
            
            renderDoc(randomDoc, docContent);
            
            setTimeout(() => {
                scrollToLine(randomDoc, insertIndex);
                const lines = document.getElementById(randomDoc).querySelectorAll('.doc-line');
                if (lines[insertIndex]) {
                    lines[insertIndex].classList.add('added');
                    setTimeout(() => {
                        lines[insertIndex].classList.remove('added');
                    }, 2000);
                }
            }, 50);
        } else if (docContent.length > 3) {
            // Remove documentation line
            const deleteIndex = Math.floor(Math.random() * docContent.length);
            const lines = document.getElementById(randomDoc).querySelectorAll('.doc-line');
            
            scrollToLine(randomDoc, deleteIndex);
            
            if (lines[deleteIndex]) {
                lines[deleteIndex].classList.add('deleted');
                
                setTimeout(() => {
                    docContent.splice(deleteIndex, 1);
                    renderDoc(randomDoc, docContent);
                }, 1000);
            }
        }
    } else {
        // Update main code
        if (Math.random() < 0.7) {
            // Add code line
            const newLine = newCodeLines[Math.floor(Math.random() * newCodeLines.length)];
            const insertIndex = Math.floor(Math.random() * (codeContent.length + 1));
            codeContent.splice(insertIndex, 0, newLine);
            
            renderCode();
            
            setTimeout(() => {
                scrollToLine('codeContent', insertIndex);
                const lines = document.getElementById('codeContent').querySelectorAll('.code-line');
                if (lines[insertIndex]) {
                    lines[insertIndex].classList.add('added');
                    setTimeout(() => {
                        lines[insertIndex].classList.remove('added');
                    }, 2000);
                }
            }, 50);
        } else if (codeContent.length > 10) {
            // Modify existing line
            const modifyIndex = Math.floor(Math.random() * codeContent.length);
            const originalLine = codeContent[modifyIndex];
            
            scrollToLine('codeContent', modifyIndex);
            
            const modifications = [
                originalLine + '  # Updated',
                originalLine.replace(/matrix/g, 'data'),
                originalLine.replace(/rank/g, 'components'),
                originalLine.replace(/gradient/g, 'grad')
            ];
            
            codeContent[modifyIndex] = modifications[Math.floor(Math.random() * modifications.length)];
            
            renderCode();
            
            setTimeout(() => {
                const lines = document.getElementById('codeContent').querySelectorAll('.code-line');
                if (lines[modifyIndex]) {
                    lines[modifyIndex].classList.add('modified');
                    setTimeout(() => {
                        lines[modifyIndex].classList.remove('modified');
                    }, 2000);
                }
            }, 50);
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
