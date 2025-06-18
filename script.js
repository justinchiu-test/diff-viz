// Simple diff visualization implementation

class DiffVisualizer {
    constructor() {
        this.currentTimestep = 0;
        this.currentPhase = 'loading';
        this.isRunning = false;
        
        // DOM elements
        this.codebankContent = document.getElementById('codebankContent');
        this.solutionPrevContent = document.getElementById('solutionPrevContent');
        this.solutionNextContent = document.getElementById('solutionNextContent');
        this.codebankFileName = document.getElementById('codebankFileName');
        this.solutionPrevFileName = document.getElementById('solutionPrevFileName');
        this.solutionNextFileName = document.getElementById('solutionNextFileName');
        this.currentPhaseEl = document.getElementById('currentPhase');
        this.timestepCountEl = document.getElementById('timestepCount');
        this.phaseIndicator = document.querySelector('.phase-indicator');
        
        this.maxTimesteps = 3; // We have time0, time1, time2
        
        // Real data from Librarian viz_data
        this.sampleData = {
            time0: {
                codebank_prev: `
import sys
from collections import deque
import heapq`,
                codebank_next: `
import sys
from collections import deque
import heapq
def compute_degrees(n, edges):

    occ = [0] * n
    for (u, v) in edges:
        occ[u] += 1
        occ[v] += 1
    return occ

def find_node_with_degree_at_least(occ, k):

    for (i, d) in enumerate(occ):
        if d >= k:
            return i
    return -1

def find_leaves(occ):

    return [i for (i, d) in enumerate(occ) if d == 1]

def assign_incident_labels(edges, node, labels, start):

    for (idx, (u, v)) in enumerate(edges):
        if labels[idx] == -1 and (u == node or v == node):
            labels[idx] = start
            start += 1
    return start

def fill_remaining_labels(labels, start):

    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = start
            start += 1
    return labels

`,
                solution_prev: `n = int(input())
occ = [0 for i in range(n)]
graph = [[0,0] for i in range(n-1)]
for i in range(n-1):
    x, y = map(int,input().split())
    occ[x-1]+=1
    occ[y-1]+=1
    graph[i][0] = x-1
    graph[i][1] = y-1
    
fin = [-1 for i in range(n-1)]
for i in range(n):
    if occ[i] >= 3 :
        var = 0
        for j in range(n-1):
            if graph[j][0] == i or graph[j][1] == i:
                fin[j] = var
                var += 1
        break
else:
    var = 0
    for i in range(n):
        if var > 1:
            break
        if occ[i] == 1:
            for j in range(n-1):
                if graph[j][0] == i or graph[j][1] == i:
                    fin[j] = var
                    var += 1
                    break
for i in fin:
    if n == 2:
        print(0)
        break
    if i == -1:
        print(var)
        var += 1
    else:
        print(i)`,
                solution_next: `from codebank import *

from codebank import *

def main():
    import sys
    data = sys.stdin
    n = int(data.readline())
    edges = []
    for _ in range(n-1):
        u, v = map(int, data.readline().split())
        edges.append((u-1, v-1))
    labels = [-1] * (n-1)
    occ = compute_degrees(n, edges)
    center = find_node_with_degree_at_least(occ, 3)
    var = 0
    if center != -1:
        var = assign_incident_labels(edges, center, labels, var)
    else:
        leaves = find_leaves(occ)
        for node in leaves[:2]:
            var = assign_incident_labels(edges, node, labels, var)
    labels = fill_remaining_labels(labels, var)
    print('\\n'.join(str(x) for x in labels))

if __name__ == "__main__":
    main()`
            },
            time1: {
                codebank_prev: `
import sys
from collections import deque
import heapq
def compute_degrees(n, edges):

    occ = [0] * n
    for (u, v) in edges:
        occ[u] += 1
        occ[v] += 1
    return occ

def find_node_with_degree_at_least(occ, k):

    for (i, d) in enumerate(occ):
        if d >= k:
            return i
    return -1

def find_leaves(occ):

    return [i for (i, d) in enumerate(occ) if d == 1]

def assign_incident_labels(edges, node, labels, start):

    for (idx, (u, v)) in enumerate(edges):
        if labels[idx] == -1 and (u == node or v == node):
            labels[idx] = start
            start += 1
    return start

def fill_remaining_labels(labels, start):

    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = start
            start += 1
    return labels

`,
                codebank_next: `
import sys
from collections import deque
import heapq
from typing import List
def compute_degrees(n, edges):

    occ = [0] * n
    for (u, v) in edges:
        occ[u] += 1
        occ[v] += 1
    return occ

def find_node_with_degree_at_least(occ, k):

    for (i, d) in enumerate(occ):
        if d >= k:
            return i
    return -1

def find_leaves(occ):

    return [i for (i, d) in enumerate(occ) if d == 1]

def assign_incident_labels(edges, node, labels, start):

    for (idx, (u, v)) in enumerate(edges):
        if labels[idx] == -1 and (u == node or v == node):
            labels[idx] = start
            start += 1
    return start

def fill_remaining_labels(labels, start):

    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = start
            start += 1
    return labels

def max_rounds_to_stabilize(arr):

    stack = []
    dp = [0] * len(arr)
    max_rounds = 0
    for (i, x) in enumerate(arr):
        max_dp = 0
        while stack and arr[stack[-1]] < x:
            max_dp = max(max_dp, dp[stack.pop()])
        if stack:
            dp[i] = max_dp + 1
            max_rounds = max(max_rounds, dp[i])
        stack.append(i)
    return max_rounds

`,
                solution_prev: `n, t = int(input()), list(map(int, input().split()))

p, s, r = [0] * n, [0] * n, t[0]

for i in range(n - 1):

    j = i + 1

    x = t[j]

    if x > r: r = x

    else:

        while t[i] < x: s[j], i = max(s[j], s[i]), p[i]

        p[j] = i

        s[j] += 1

print(max(s))



# Made By Mostafa_Khaled`,
                solution_next: `from codebank import *

from codebank import *

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    print(max_rounds_to_stabilize(arr))

if __name__ == "__main__":
    main()`
            },
            time2: {
                codebank_prev: `
import sys
from collections import deque
import heapq
from typing import List
def compute_degrees(n, edges):

    occ = [0] * n
    for (u, v) in edges:
        occ[u] += 1
        occ[v] += 1
    return occ

def find_node_with_degree_at_least(occ, k):

    for (i, d) in enumerate(occ):
        if d >= k:
            return i
    return -1

def find_leaves(occ):

    return [i for (i, d) in enumerate(occ) if d == 1]

def assign_incident_labels(edges, node, labels, start):

    for (idx, (u, v)) in enumerate(edges):
        if labels[idx] == -1 and (u == node or v == node):
            labels[idx] = start
            start += 1
    return start

def fill_remaining_labels(labels, start):

    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = start
            start += 1
    return labels

def max_rounds_to_stabilize(arr):

    stack = []
    dp = [0] * len(arr)
    max_rounds = 0
    for (i, x) in enumerate(arr):
        max_dp = 0
        while stack and arr[stack[-1]] < x:
            max_dp = max(max_dp, dp[stack.pop()])
        if stack:
            dp[i] = max_dp + 1
            max_rounds = max(max_rounds, dp[i])
        stack.append(i)
    return max_rounds

`,
                codebank_next: `
import heapq
import sys
from collections import defaultdict
from typing import List
from collections import deque
def compute_degrees(n, edges):

    occ = [0] * n
    for (u, v) in edges:
        occ[u] += 1
        occ[v] += 1
    return occ

def find_node_with_degree_at_least(occ, k):

    for (i, d) in enumerate(occ):
        if d >= k:
            return i
    return -1

def find_leaves(occ):

    return [i for (i, d) in enumerate(occ) if d == 1]

def assign_incident_labels(edges, node, labels, start):

    for (idx, (u, v)) in enumerate(edges):
        if labels[idx] == -1 and (u == node or v == node):
            labels[idx] = start
            start += 1
    return start

def fill_remaining_labels(labels, start):

    for i in range(len(labels)):
        if labels[i] == -1:
            labels[i] = start
            start += 1
    return labels

def max_rounds_to_stabilize(arr):

    stack = []
    dp = [0] * len(arr)
    max_rounds = 0
    for (i, x) in enumerate(arr):
        max_dp = 0
        while stack and arr[stack[-1]] < x:
            max_dp = max(max_dp, dp[stack.pop()])
        if stack:
            dp[i] = max_dp + 1
            max_rounds = max(max_rounds, dp[i])
        stack.append(i)
    return max_rounds

def build_graph(n, edges):

    graph = [[] for _ in range(n)]
    for (u, v) in edges:
        graph[u - 1].append(v - 1)
        graph[v - 1].append(u - 1)
    return graph

def dfs_subtree_sizes(node, parent, graph, sizes):

    total = 1
    for nei in graph[node]:
        if nei != parent:
            total += dfs_subtree_sizes(nei, node, graph, sizes)
    sizes[node] = total
    return total

def compute_subtree_sizes(graph, n, root=0):

    sizes = [0] * n
    dfs_subtree_sizes(root, -1, graph, sizes)
    return sizes

def count_removable_edges(sizes):

    return sum((1 for size in sizes[1:] if size % 2 == 0))

`,
                solution_prev: `from collections import  defaultdict
import threading
from sys import stdin,setrecursionlimit
setrecursionlimit(300000)
input=stdin.readline

def dfs(node,g,par,sz):
	for i in g[node]:
		if i!=par:
			sz[node]+=dfs(i,g,node,sz)
	return sz[node]+1
def main():
	n=int(input())
	if n%2!=0:
		print(-1)
		exit(0)
	g=defaultdict(list)
	for i in range(n-1):
		x,y=map(int,input().strip().split())
		g[x-1].append(y-1)
		g[y-1].append(x-1)

	sz=[0]*(n)
	tt=[]
	dfs(0,g,-1,sz)
	res=0
	# print(sz)
	for i in range(1,n):
		if sz[i]%2!=0:
			res+=1
	print(res)

threading.stack_size(10 ** 8)
t = threading.Thread(target=main)
t.start()
t.join()`,
                solution_next: `from codebank import *

from codebank import *

import sys

def main():
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    n = int(input())
    if n & 1:
        print(-1)
        return
    edges = [tuple(map(int, input().split())) for _ in range(n-1)]
    graph = build_graph(n, edges)
    sizes = compute_subtree_sizes(graph, n)
    print(count_removable_edges(sizes))

if __name__ == "__main__":
    main()`
            }
        };
    }

    async start() {
        this.isRunning = true;
        this.updateStatus('loading', 'Initializing...');
        
        while (this.isRunning && this.currentTimestep < this.maxTimesteps) {
            await this.processTimestep(this.currentTimestep);
            this.currentTimestep++;
            await this.delay(200); // Very short pause between timesteps
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
        this.solutionPrevFileName.textContent = `solution${timestep + 1}_prev.py`;
        this.solutionNextFileName.textContent = `solution${timestep + 1}_next.py`;
        
        // Show initial state
        this.displayCode(codebankPrev, this.codebankContent);
        this.displayCode(solutionPrev, this.solutionPrevContent);
        this.solutionNextContent.innerHTML = ''; // Start empty
        await this.delay(1000);
        
        // Phase 1: Animate codebank changes
        this.updateStatus('codebank', 'Animating codebank changes');
        await this.animateDiff(codebankPrev, codebankNext, this.codebankContent);
        await this.delay(500);
        
        // Phase 2: Write solution_next line by line
        this.updateStatus('solution', 'Writing solution_next.py');
        await this.writeCodeLineByLine(solutionNext, this.solutionNextContent);
        await this.delay(200);
    }

    displayCode(content, container) {
        const lines = content.split('\n');
        container.innerHTML = lines.map((line, i) => `
            <div class="code-line" data-line="${i + 1}">
                <div class="line-number">${i + 1}</div>
                <div class="line-content">${this.applySyntaxHighlighting(line)}</div>
            </div>
        `).join('');
    }

    applySyntaxHighlighting(code) {
        const escaped = this.escapeHtml(code);
        const tokens = this.tokenize(escaped);
        return this.renderTokens(tokens);
    }

    tokenize(code) {
        const tokens = [];
        let i = 0;
        
        while (i < code.length) {
            let matched = false;
            
            // Check for triple-quoted strings first (highest priority)
            const tripleQuoteMatch = code.slice(i).match(/^("""[\s\S]*?"""|'''[\s\S]*?''')/);
            if (tripleQuoteMatch) {
                tokens.push({ type: 'docstring', value: tripleQuoteMatch[1] });
                i += tripleQuoteMatch[1].length;
                matched = true;
                continue;
            }
            
            // Check for comments
            const commentMatch = code.slice(i).match(/^#.*$/m);
            if (commentMatch) {
                tokens.push({ type: 'comment', value: commentMatch[0] });
                i += commentMatch[0].length;
                matched = true;
                continue;
            }
            
            // Check for string literals
            const stringMatch = code.slice(i).match(/^(["'])((?:[^\\]|\\.)*)(\1)/);
            if (stringMatch) {
                tokens.push({ type: 'string', value: stringMatch[0] });
                i += stringMatch[0].length;
                matched = true;
                continue;
            }
            
            // Check for numbers
            const numberMatch = code.slice(i).match(/^\d+\.?\d*/);
            if (numberMatch) {
                tokens.push({ type: 'number', value: numberMatch[0] });
                i += numberMatch[0].length;
                matched = true;
                continue;
            }
            
            // Check for keywords
            const keywordMatch = code.slice(i).match(/^(def|class|import|from|if|else|elif|for|while|return|try|except|finally|with|as|in|and|or|not|is|lambda|yield|async|await|pass|break|continue|raise|assert|del|global|nonlocal|True|False|None)\b/);
            if (keywordMatch) {
                tokens.push({ type: 'keyword', value: keywordMatch[1] });
                i += keywordMatch[1].length;
                matched = true;
                continue;
            }
            
            // Check for function calls
            const functionMatch = code.slice(i).match(/^([a-zA-Z_][a-zA-Z0-9_]*)(?=\s*\()/);
            if (functionMatch) {
                tokens.push({ type: 'function', value: functionMatch[1] });
                i += functionMatch[1].length;
                matched = true;
                continue;
            }
            
            // If no pattern matched, add the character as plain text
            if (!matched) {
                tokens.push({ type: 'plain', value: code[i] });
                i++;
            }
        }
        
        return tokens;
    }

    renderTokens(tokens) {
        return tokens.map(token => {
            switch (token.type) {
                case 'docstring':
                    return `<span class="docstring">${token.value}</span>`;
                case 'string':
                    return `<span class="string">${token.value}</span>`;
                case 'comment':
                    return `<span class="comment">${token.value}</span>`;
                case 'keyword':
                    return `<span class="keyword">${token.value}</span>`;
                case 'number':
                    return `<span class="number">${token.value}</span>`;
                case 'function':
                    return `<span class="function">${token.value}</span>`;
                default:
                    return token.value;
            }
        }).join('');
    }

    async animateDiff(prevContent, nextContent, container) {
        const prevLines = prevContent.split('\n');
        const nextLines = nextContent.split('\n');
        
        // Find added lines and animate them one by one
        await this.animateLineByLine(prevLines, nextLines, container);
    }

    async animateLineByLine(prevLines, nextLines, container) {
        // Start by showing the previous version
        this.displayCode(prevLines.join('\n'), container);
        await this.delay(500); // Brief pause to show the starting state
        
        // Find actual differences using proper diff algorithm
        const changes = this.calculateLineDiff(prevLines, nextLines);
        
        // Build the content incrementally by adding new lines
        let currentContent = [...prevLines];
        
        for (const change of changes) {
            if (change.type === 'add') {
                // Insert the new line at the correct position
                currentContent.splice(change.insertIndex, 0, change.content);
                
                // Update the display with the new content
                this.displayCode(currentContent.join('\n'), container);
                
                // Find and highlight the newly added line
                const lineEl = container.querySelector(`[data-line="${change.insertIndex + 1}"]`);
                if (lineEl) {
                    // Scroll to the line
                    this.scrollToLine(change.insertIndex + 1, container);
                    
                    // Add green highlight
                    lineEl.classList.add('added');
                    
                    // Remove highlight after short delay
                    setTimeout(() => {
                        lineEl.classList.remove('added');
                    }, 600);
                    
                    // Short delay before next line
                    await this.delay(150);
                }
            }
        }
    }

    calculateLineDiff(prevLines, nextLines) {
        const changes = [];
        const prevSet = new Set(prevLines);
        
        // Find where to insert new lines by comparing sequences
        let prevIndex = 0;
        
        for (let nextIndex = 0; nextIndex < nextLines.length; nextIndex++) {
            const line = nextLines[nextIndex];
            
            if (prevSet.has(line)) {
                // This line exists in prev, advance the prev pointer
                while (prevIndex < prevLines.length && prevLines[prevIndex] !== line) {
                    prevIndex++;
                }
                prevIndex++;
            } else {
                // This is a new line, add it to changes
                changes.push({
                    type: 'add',
                    insertIndex: prevIndex,
                    content: line
                });
                prevIndex++; // Account for the line we're inserting
            }
        }
        
        return changes;
    }

    async writeCodeLineByLine(content, container) {
        const lines = content.split('\n');
        
        for (let i = 0; i < lines.length; i++) {
            // Build content up to current line
            const currentContent = lines.slice(0, i + 1).join('\n');
            this.displayCode(currentContent, container);
            
            // Highlight the newly added line
            const lineEl = container.querySelector(`[data-line="${i + 1}"]`);
            if (lineEl) {
                // Scroll to the line
                this.scrollToLine(i + 1, container);
                
                // Add green highlight
                lineEl.classList.add('added');
                
                // Remove highlight after short delay
                setTimeout(() => {
                    lineEl.classList.remove('added');
                }, 600);
                
                // Short delay before next line
                await this.delay(100);
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