// Simple diff visualization implementation

class DiffVisualizer {
    constructor() {
        this.currentTimestep = 0;
        this.currentPhase = 'loading';
        this.isRunning = false;
        
        // DOM elements
        this.codebankContent = document.getElementById('codebankContent');
        this.solutionContent = document.getElementById('solutionContent');
        this.codebankFileName = document.getElementById('codebankFileName');
        this.solutionFileName = document.getElementById('solutionFileName');
        this.currentPhaseEl = document.getElementById('currentPhase');
        this.timestepCountEl = document.getElementById('timestepCount');
        this.phaseIndicator = document.querySelector('.phase-indicator');
        
        this.maxTimesteps = 3; // We have time0, time1, time2
        
        // Embedded sample data
        this.sampleData = {
            time0: {
                codebank_prev: `#!/usr/bin/env python3
"""
Basic matrix operations library.
"""

import numpy as np

def add_matrices(a, b):
    """Add two matrices."""
    return a + b

def multiply_matrices(a, b):
    """Multiply two matrices."""
    return np.dot(a, b)`,
                codebank_next: `#!/usr/bin/env python3
"""
Basic matrix operations library.
Enhanced with input validation.
"""

import numpy as np

def validate_input(matrix):
    """Validate matrix input."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")
    return True

def add_matrices(a, b):
    """Add two matrices with validation."""
    validate_input(a)
    validate_input(b)
    return a + b

def multiply_matrices(a, b):
    """Multiply two matrices with validation."""
    validate_input(a)
    validate_input(b)
    return np.dot(a, b)`,
                solution_prev: `import numpy as np
from codebank import add_matrices, multiply_matrices

# Test basic operations
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

result_add = add_matrices(a, b)
result_mult = multiply_matrices(a, b)

print("Addition result:", result_add)
print("Multiplication result:", result_mult)`,
                solution_next: `import numpy as np
from codebank import add_matrices, multiply_matrices

# Test enhanced operations with error handling
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

try:
    result_add = add_matrices(a, b)
    result_mult = multiply_matrices(a, b)
    
    print("Addition result:", result_add)
    print("Multiplication result:", result_mult)
    
    # Test with invalid input
    invalid_input = [[1, 2], [3, 4]]  # Regular list, not numpy array
    result_invalid = add_matrices(a, invalid_input)
    
except TypeError as e:
    print(f"Error caught: {e}")
    print("Validation is working correctly!")`
            },
            time1: {
                codebank_prev: `#!/usr/bin/env python3
"""
Basic matrix operations library.
Enhanced with input validation.
"""

import numpy as np

def validate_input(matrix):
    """Validate matrix input."""
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input must be a numpy array")
    return True

def add_matrices(a, b):
    """Add two matrices with validation."""
    validate_input(a)
    validate_input(b)
    return a + b

def multiply_matrices(a, b):
    """Multiply two matrices with validation."""
    validate_input(a)
    validate_input(b)
    return np.dot(a, b)`,
                codebank_next: `#!/usr/bin/env python3
"""
Advanced matrix operations library.
Enhanced with input validation and logging.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_input(matrix):
    """Validate matrix input with logging."""
    if not isinstance(matrix, np.ndarray):
        logger.error("Invalid input type: %s", type(matrix))
        raise TypeError("Input must be a numpy array")
    logger.debug("Input validation passed")
    return True

def add_matrices(a, b):
    """Add two matrices with validation."""
    validate_input(a)
    validate_input(b)
    logger.info("Adding matrices of shapes %s and %s", a.shape, b.shape)
    return a + b

def multiply_matrices(a, b):
    """Multiply two matrices with validation."""
    validate_input(a)
    validate_input(b)
    logger.info("Multiplying matrices of shapes %s and %s", a.shape, b.shape)
    return np.dot(a, b)

def transpose_matrix(matrix):
    """Transpose a matrix."""
    validate_input(matrix)
    logger.info("Transposing matrix of shape %s", matrix.shape)
    return matrix.T`,
                solution_prev: `import numpy as np
from codebank import add_matrices, multiply_matrices

# Test enhanced operations with error handling
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

try:
    result_add = add_matrices(a, b)
    result_mult = multiply_matrices(a, b)
    
    print("Addition result:", result_add)
    print("Multiplication result:", result_mult)
    
except TypeError as e:
    print(f"Error caught: {e}")`,
                solution_next: `import numpy as np
import logging
from codebank import add_matrices, multiply_matrices, transpose_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)

# Test enhanced operations with logging
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

try:
    result_add = add_matrices(a, b)
    result_mult = multiply_matrices(a, b)
    result_transpose = transpose_matrix(a)
    
    print("Addition result:", result_add)
    print("Multiplication result:", result_mult)
    print("Transpose result:", result_transpose)
    
    # Performance test
    large_a = np.random.randn(100, 100)
    large_b = np.random.randn(100, 100)
    
    large_result = multiply_matrices(large_a, large_b)
    print(f"Large matrix multiplication completed: {large_result.shape}")
    
except TypeError as e:
    print(f"Error caught: {e}")`
            },
            time2: {
                codebank_prev: `#!/usr/bin/env python3
"""
Advanced matrix operations library.
Enhanced with input validation and logging.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

def validate_input(matrix):
    """Validate matrix input with logging."""
    if not isinstance(matrix, np.ndarray):
        logger.error("Invalid input type: %s", type(matrix))
        raise TypeError("Input must be a numpy array")
    logger.debug("Input validation passed")
    return True

def add_matrices(a, b):
    """Add two matrices with validation."""
    validate_input(a)
    validate_input(b)
    logger.info("Adding matrices of shapes %s and %s", a.shape, b.shape)
    return a + b

def multiply_matrices(a, b):
    """Multiply two matrices with validation."""
    validate_input(a)
    validate_input(b)
    logger.info("Multiplying matrices of shapes %s and %s", a.shape, b.shape)
    return np.dot(a, b)

def transpose_matrix(matrix):
    """Transpose a matrix."""
    validate_input(matrix)
    logger.info("Transposing matrix of shape %s", matrix.shape)
    return matrix.T`,
                codebank_next: `#!/usr/bin/env python3
"""
Professional matrix operations library.
Enhanced with input validation, logging, and performance optimization.
"""

import numpy as np
import logging
from typing import Union
import time

logger = logging.getLogger(__name__)

def validate_input(matrix: np.ndarray) -> bool:
    """Validate matrix input with comprehensive checks."""
    if not isinstance(matrix, np.ndarray):
        logger.error("Invalid input type: %s", type(matrix))
        raise TypeError("Input must be a numpy array")
    
    if matrix.size == 0:
        raise ValueError("Matrix cannot be empty")
    
    if not np.isfinite(matrix).all():
        raise ValueError("Matrix contains invalid values (NaN or Inf)")
    
    logger.debug("Input validation passed for matrix shape %s", matrix.shape)
    return True

def add_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Add two matrices with validation and timing."""
    start_time = time.time()
    validate_input(a)
    validate_input(b)
    
    if a.shape != b.shape:
        raise ValueError(f"Matrix shapes don't match: {a.shape} vs {b.shape}")
    
    result = a + b
    elapsed = time.time() - start_time
    logger.info("Added matrices in %.4f seconds", elapsed)
    return result

def multiply_matrices(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Multiply two matrices with validation and timing."""
    start_time = time.time()
    validate_input(a)
    validate_input(b)
    
    result = np.dot(a, b)
    elapsed = time.time() - start_time
    logger.info("Multiplied matrices in %.4f seconds", elapsed)
    return result

def transpose_matrix(matrix: np.ndarray) -> np.ndarray:
    """Transpose a matrix with validation."""
    validate_input(matrix)
    logger.info("Transposing matrix of shape %s", matrix.shape)
    return matrix.T

def matrix_norm(matrix: np.ndarray, ord: Union[int, str] = 'fro') -> float:
    """Calculate matrix norm."""
    validate_input(matrix)
    norm_value = np.linalg.norm(matrix, ord=ord)
    logger.info("Calculated %s norm: %.6f", ord, norm_value)
    return norm_value`,
                solution_prev: `import numpy as np
import logging
from codebank import add_matrices, multiply_matrices, transpose_matrix

# Configure logging
logging.basicConfig(level=logging.INFO)

# Test enhanced operations with logging
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])

try:
    result_add = add_matrices(a, b)
    result_mult = multiply_matrices(a, b)
    result_transpose = transpose_matrix(a)
    
    print("Addition result:", result_add)
    print("Multiplication result:", result_mult)
    print("Transpose result:", result_transpose)
    
except TypeError as e:
    print(f"Error caught: {e}")`,
                solution_next: `import numpy as np
import logging
from codebank import add_matrices, multiply_matrices, transpose_matrix, matrix_norm

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Comprehensive testing suite
def run_tests():
    """Run comprehensive matrix operation tests."""
    print("=== Matrix Operations Test Suite ===")
    
    # Basic operations
    a = np.array([[1, 2], [3, 4]])
    b = np.array([[5, 6], [7, 8]])
    
    try:
        result_add = add_matrices(a, b)
        result_mult = multiply_matrices(a, b)
        result_transpose = transpose_matrix(a)
        norm_value = matrix_norm(a)
        
        print("Addition result:", result_add)
        print("Multiplication result:", result_mult)
        print("Transpose result:", result_transpose)
        print(f"Matrix norm: {norm_value:.6f}")
        
        # Performance benchmark
        print("\\n=== Performance Benchmark ===")
        sizes = [50, 100, 200]
        
        for size in sizes:
            large_a = np.random.randn(size, size)
            large_b = np.random.randn(size, size)
            
            large_result = multiply_matrices(large_a, large_b)
            print(f"Completed {size}x{size} matrix multiplication")
        
        # Error handling test
        print("\\n=== Error Handling Test ===")
        try:
            invalid_matrix = np.array([[1, np.inf], [3, 4]])
            add_matrices(a, invalid_matrix)
        except ValueError as e:
            print(f"Successfully caught error: {e}")
            
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    run_tests()`
            }
        };
    }

    async start() {
        this.isRunning = true;
        this.updateStatus('loading', 'Initializing...');
        
        while (this.isRunning && this.currentTimestep < this.maxTimesteps) {
            await this.processTimestep(this.currentTimestep);
            this.currentTimestep++;
            await this.delay(2000); // Pause between timesteps
        }
        
        this.updateStatus('complete', 'Animation Complete');
    }

    async processTimestep(timestep) {
        this.timestepCountEl.textContent = timestep + 1;
        
        const timestepKey = `time${timestep}`;
        
        // Load initial content for both panels
        this.updateStatus('loading', 'Loading timestep files...');
        
        const codebankPrev = this.sampleData[timestepKey].codebank_prev;
        const codebankNext = this.sampleData[timestepKey].codebank_next;
        const solutionPrev = this.sampleData[timestepKey].solution_prev;
        const solutionNext = this.sampleData[timestepKey].solution_next;
        
        // Update file names
        this.codebankFileName.textContent = 'codebank.py';
        this.solutionFileName.textContent = `solution${timestep + 1}.py`;
        
        // Show initial state (prev versions)
        this.displayCode(codebankPrev, this.codebankContent);
        this.displayCode(solutionPrev, this.solutionContent);
        await this.delay(1000);
        
        // Phase 1: Animate codebank changes
        this.updateStatus('codebank', 'Animating codebank changes');
        await this.animateDiff(codebankPrev, codebankNext, this.codebankContent);
        await this.delay(1500);
        
        // Phase 2: Animate solution changes
        this.updateStatus('solution', 'Animating solution changes');
        await this.animateDiff(solutionPrev, solutionNext, this.solutionContent);
        await this.delay(1000);
    }

    displayCode(content, container) {
        const lines = content.split('\n');
        container.innerHTML = lines.map((line, i) => `
            <div class="code-line" data-line="${i + 1}">
                <div class="line-number">${i + 1}</div>
                <div class="line-content">${this.escapeHtml(line)}</div>
            </div>
        `).join('');
    }

    async animateDiff(prevContent, nextContent, container) {
        const prevLines = prevContent.split('\n');
        const nextLines = nextContent.split('\n');
        
        // Find added lines and animate them one by one
        await this.animateLineByLine(prevLines, nextLines, container);
    }

    async animateLineByLine(prevLines, nextLines, container) {
        // Simple line-by-line diff: find which lines are new
        const maxLines = Math.max(prevLines.length, nextLines.length);
        
        // First, display the final content
        this.displayCode(nextLines.join('\n'), container);
        
        // Then animate the added lines
        for (let i = 0; i < nextLines.length; i++) {
            const currentLine = nextLines[i];
            const isNewLine = i >= prevLines.length || currentLine !== prevLines[i];
            
            if (isNewLine) {
                const lineEl = container.querySelector(`[data-line="${i + 1}"]`);
                if (lineEl) {
                    // Scroll to the line
                    this.scrollToLine(i + 1, container);
                    
                    // Add green highlight
                    lineEl.classList.add('added');
                    
                    // Remove highlight after short delay
                    setTimeout(() => {
                        lineEl.classList.remove('added');
                    }, 600); // Even faster transition
                    
                    // Very short delay before next line
                    await this.delay(150);
                }
            }
        }
    }

    calculateDiff(prevLines, nextLines) {
        const changes = [];
        let prevIndex = 0;
        let nextIndex = 0;
        
        // Simple diff algorithm
        while (prevIndex < prevLines.length || nextIndex < nextLines.length) {
            if (prevIndex >= prevLines.length) {
                // Only additions left
                changes.push({
                    type: 'add',
                    line: nextIndex + 1,
                    content: nextLines[nextIndex]
                });
                nextIndex++;
            } else if (nextIndex >= nextLines.length) {
                // Only deletions left
                changes.push({
                    type: 'remove',
                    line: prevIndex + 1,
                    content: prevLines[prevIndex]
                });
                prevIndex++;
            } else if (prevLines[prevIndex] === nextLines[nextIndex]) {
                // Lines match, continue
                prevIndex++;
                nextIndex++;
            } else {
                // Lines differ - for simplicity, treat as modification
                changes.push({
                    type: 'modify',
                    line: prevIndex + 1,
                    oldContent: prevLines[prevIndex],
                    newContent: nextLines[nextIndex]
                });
                prevIndex++;
                nextIndex++;
            }
        }
        
        return changes;
    }

    async animateChange(change, container) {
        const lineEl = container.querySelector(`[data-line="${change.line}"]`);
        if (!lineEl) return;
        
        // Scroll to the change
        this.scrollToLine(change.line, container);
        
        // Apply visual change
        switch (change.type) {
            case 'add':
                lineEl.classList.add('added');
                lineEl.querySelector('.line-content').textContent = change.content;
                break;
            case 'remove':
                lineEl.classList.add('removed');
                break;
            case 'modify':
                lineEl.classList.add('modified');
                lineEl.querySelector('.line-content').textContent = change.newContent;
                break;
        }
        
        // Remove highlight after delay
        setTimeout(() => {
            lineEl.classList.remove('added', 'removed', 'modified');
        }, 2000);
    }

    scrollToLine(lineNumber, container) {
        const lineEl = container.querySelector(`[data-line="${lineNumber}"]`);
        if (lineEl) {
            lineEl.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'center' 
            });
        }
    }

    updateStatus(phase, message) {
        this.currentPhase = phase;
        this.currentPhaseEl.textContent = message;
        
        // Update phase indicator
        this.phaseIndicator.className = `phase-indicator phase-${phase}`;
    }


    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Start visualization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    const visualizer = new DiffVisualizer();
    visualizer.start();
});