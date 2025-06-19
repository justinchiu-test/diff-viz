// Simple diff visualization implementation

class DiffVisualizer {
    constructor() {
        this.currentTimestep = 0;
        this.currentPhase = 'loading';
        this.isRunning = false;
        
        // DOM elements
        this.libraryContent = document.getElementById('libraryContent');
        this.solutionContent = document.getElementById('solutionContent');
        this.libraryFileName = document.getElementById('libraryFileName');
        this.solutionFileName = document.getElementById('solutionFileName');
        this.currentPhaseEl = document.getElementById('currentPhase');
        this.timestepCountEl = document.getElementById('timestepCount');
        this.phaseIndicator = document.querySelector('.phase-indicator');
        
        this.maxTimesteps = 3; // We have time0, time1, time2
        
        // Real data from Librarian viz_data
        this.sampleData = {
            time0: {
                library_prev: `
import heapq
from collections import deque
import sys`,
                library_next: `
from collections import deque
import heapq
import sys
def read_tree():
    n = int(input())
    edges = []
    degrees = [0] * n
    for _ in range(n - 1):
        (u, v) = map(int, input().split())
        u -= 1
        v -= 1
        edges.append((u, v))
        degrees[u] += 1
        degrees[v] += 1
    return (n, edges, degrees)

def find_node_by_degree(degrees, threshold):
    for (i, d) in enumerate(degrees):
        if d >= threshold:
            return i
    return None

def label_edges_at_node(edges, node, fin, start):
    lab = start
    for (idx, (u, v)) in enumerate(edges):
        if u == node or v == node:
            fin[idx] = lab
            lab += 1
    return lab

def find_leaves(degrees):
    return [i for (i, d) in enumerate(degrees) if d == 1]

def label_edges_at_leaves(edges, leaves, fin, start):
    lab = start
    cnt = 0
    for leaf in leaves:
        if cnt >= 2:
            break
        for (idx, (u, v)) in enumerate(edges):
            if leaf == u or leaf == v:
                fin[idx] = lab
                lab += 1
                cnt += 1
                break
    return lab

def label_remaining(fin, start):
    lab = start
    for i in range(len(fin)):
        if fin[i] == -1:
            fin[i] = lab
            lab += 1
    return fin
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
                solution_next: `from library import *

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
                library_prev: `
from collections import deque
import heapq
import sys
def read_tree():
    n = int(input())
    edges = []
    degrees = [0] * n
    for _ in range(n - 1):
        (u, v) = map(int, input().split())
        u -= 1
        v -= 1
        edges.append((u, v))
        degrees[u] += 1
        degrees[v] += 1
    return (n, edges, degrees)

def find_node_by_degree(degrees, threshold):
    for (i, d) in enumerate(degrees):
        if d >= threshold:
            return i
    return None

def label_edges_at_node(edges, node, fin, start):
    lab = start
    for (idx, (u, v)) in enumerate(edges):
        if u == node or v == node:
            fin[idx] = lab
            lab += 1
    return lab

def find_leaves(degrees):
    return [i for (i, d) in enumerate(degrees) if d == 1]

def label_edges_at_leaves(edges, leaves, fin, start):
    lab = start
    cnt = 0
    for leaf in leaves:
        if cnt >= 2:
            break
        for (idx, (u, v)) in enumerate(edges):
            if leaf == u or leaf == v:
                fin[idx] = lab
                lab += 1
                cnt += 1
                break
    return lab

def label_remaining(fin, start):
    lab = start
    for i in range(len(fin)):
        if fin[i] == -1:
            fin[i] = lab
            lab += 1
    return fin
`,
                library_next: `
import heapq
from collections import deque
import sys
def read_tree():
    n = int(input())
    edges = []
    degrees = [0] * n
    for _ in range(n - 1):
        (u, v) = map(int, input().split())
        u -= 1
        v -= 1
        edges.append((u, v))
        degrees[u] += 1
        degrees[v] += 1
    return (n, edges, degrees)

def find_node_by_degree(degrees, threshold):
    for (i, d) in enumerate(degrees):
        if d >= threshold:
            return i
    return None

def label_edges_at_node(edges, node, fin, start):
    lab = start
    for (idx, (u, v)) in enumerate(edges):
        if u == node or v == node:
            fin[idx] = lab
            lab += 1
    return lab

def find_leaves(degrees):
    return [i for (i, d) in enumerate(degrees) if d == 1]

def label_edges_at_leaves(edges, leaves, fin, start):
    lab = start
    cnt = 0
    for leaf in leaves:
        if cnt >= 2:
            break
        for (idx, (u, v)) in enumerate(edges):
            if leaf == u or leaf == v:
                fin[idx] = lab
                lab += 1
                cnt += 1
                break
    return lab

def label_remaining(fin, start):
    lab = start
    for i in range(len(fin)):
        if fin[i] == -1:
            fin[i] = lab
            lab += 1
    return fin

def read_sequence():
    n = int(input())
    return list(map(int, input().split()))

def compute_kill_steps(seq):
    stack = []
    max_steps = 0
    for x in seq:
        steps = 0
        while stack and x >= stack[-1][0]:
            steps = max(steps, stack[-1][1])
            stack.pop()
        if stack:
            steps += 1
        else:
            steps = 0
        max_steps = max(max_steps, steps)
        stack.append((x, steps))
    return max_steps
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

print(max(s))`,
                solution_next: `from library import *

def main():
    n = int(input())
    arr = list(map(int, input().split()))
    print(compute_steps(arr))

if __name__ == "__main__":
    main()`
            },
            time2: {
                library_prev: `
import heapq
from collections import deque
import sys
def read_tree():
    n = int(input())
    edges = []
    degrees = [0] * n
    for _ in range(n - 1):
        (u, v) = map(int, input().split())
        u -= 1
        v -= 1
        edges.append((u, v))
        degrees[u] += 1
        degrees[v] += 1
    return (n, edges, degrees)

def find_node_by_degree(degrees, threshold):
    for (i, d) in enumerate(degrees):
        if d >= threshold:
            return i
    return None

def label_edges_at_node(edges, node, fin, start):
    lab = start
    for (idx, (u, v)) in enumerate(edges):
        if u == node or v == node:
            fin[idx] = lab
            lab += 1
    return lab

def find_leaves(degrees):
    return [i for (i, d) in enumerate(degrees) if d == 1]

def label_edges_at_leaves(edges, leaves, fin, start):
    lab = start
    cnt = 0
    for leaf in leaves:
        if cnt >= 2:
            break
        for (idx, (u, v)) in enumerate(edges):
            if leaf == u or leaf == v:
                fin[idx] = lab
                lab += 1
                cnt += 1
                break
    return lab

def label_remaining(fin, start):
    lab = start
    for i in range(len(fin)):
        if fin[i] == -1:
            fin[i] = lab
            lab += 1
    return fin

def read_sequence():
    n = int(input())
    return list(map(int, input().split()))

def compute_kill_steps(seq):
    stack = []
    max_steps = 0
    for x in seq:
        steps = 0
        while stack and x >= stack[-1][0]:
            steps = max(steps, stack[-1][1])
            stack.pop()
        if stack:
            steps += 1
        else:
            steps = 0
        max_steps = max(max_steps, steps)
        stack.append((x, steps))
    return max_steps
`,
                library_next: `
from collections import deque
from collections import defaultdict
import heapq
import sys
def read_tree():
    n = int(input())
    edges = []
    degrees = [0] * n
    for _ in range(n - 1):
        (u, v) = map(int, input().split())
        u -= 1
        v -= 1
        edges.append((u, v))
        degrees[u] += 1
        degrees[v] += 1
    return (n, edges, degrees)

def find_node_by_degree(degrees, threshold):
    for (i, d) in enumerate(degrees):
        if d >= threshold:
            return i
    return None

def label_edges_at_node(edges, node, fin, start):
    lab = start
    for (idx, (u, v)) in enumerate(edges):
        if u == node or v == node:
            fin[idx] = lab
            lab += 1
    return lab

def find_leaves(degrees):
    return [i for (i, d) in enumerate(degrees) if d == 1]

def label_edges_at_leaves(edges, leaves, fin, start):
    lab = start
    cnt = 0
    for leaf in leaves:
        if cnt >= 2:
            break
        for (idx, (u, v)) in enumerate(edges):
            if leaf == u or leaf == v:
                fin[idx] = lab
                lab += 1
                cnt += 1
                break
    return lab

def label_remaining(fin, start):
    lab = start
    for i in range(len(fin)):
        if fin[i] == -1:
            fin[i] = lab
            lab += 1
    return fin

def read_sequence():
    n = int(input())
    return list(map(int, input().split()))

def compute_kill_steps(seq):
    stack = []
    max_steps = 0
    for x in seq:
        steps = 0
        while stack and x >= stack[-1][0]:
            steps = max(steps, stack[-1][1])
            stack.pop()
        if stack:
            steps += 1
        else:
            steps = 0
        max_steps = max(max_steps, steps)
        stack.append((x, steps))
    return max_steps

def build_adj_list(n, edges):
    adj = [[] for _ in range(n)]
    for (u, v) in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def dfs_subtree_size(node, parent, adj, sizes):
    total = 1
    for nei in adj[node]:
        if nei != parent:
            total += dfs_subtree_size(nei, node, adj, sizes)
    sizes[node] = total
    return total

def compute_subtree_sizes(n, adj):
    sizes = [0] * n
    dfs_subtree_size(0, -1, adj, sizes)
    return sizes
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
	for i in range(1,n):
		if sz[i]%2!=0:
			res+=1
	print(res)

threading.stack_size(10 ** 8)
t = threading.Thread(target=main)
t.start()
t.join()`,
                solution_next: `from library import *

import sys

def main():
    sys.setrecursionlimit(10**7)
    input = sys.stdin.readline
    n = int(input())
    if n & 1:
        print(-1)
        return
    edges = []
    for _ in range(n-1):
        u, v = map(int, input().split())
        edges.append((u-1, v-1))
    graph = build_adj_list(n, edges)
    sizes = compute_subtree_sizes(n, graph)
    
    res = 0
    for i in range(1, n):
        if sizes[i] % 2 != 0:
            res += 1
    print(res)

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
        
        const libraryPrev = this.sampleData[timestepKey].library_prev;
        const libraryNext = this.sampleData[timestepKey].library_next;
        const solutionPrev = this.sampleData[timestepKey].solution_prev;
        const solutionNext = this.sampleData[timestepKey].solution_next;
        
        // Update file names
        this.libraryFileName.textContent = 'library.py';
        this.solutionFileName.textContent = `solution${timestep + 1}.py`;
        
        // Show initial state (prev versions)
        this.displayCode(libraryPrev, this.libraryContent);
        this.displayCode(solutionPrev, this.solutionContent);
        await this.delay(1000);
        
        // Phase 1: Animate library changes
        this.updateStatus('library', 'Updating library.py');
        await this.animateDiff(libraryPrev, libraryNext, this.libraryContent);
        await this.delay(500);
        
        // Phase 2: Animate solution changes
        this.updateStatus('solution', `Updating solution${timestep + 1}.py`);
        await this.animateDiff(solutionPrev, solutionNext, this.solutionContent);
        await this.delay(200);
    }

    displayCode(content, container, highlightedLines = []) {
        const lines = content.split('\n');
        container.innerHTML = lines.map((line, i) => {
            const lineNum = i + 1;
            const isHighlighted = highlightedLines.includes(lineNum);
            return `
                <div class="code-line ${isHighlighted ? 'deleted' : ''}" data-line="${lineNum}">
                    <div class="line-number">${lineNum}</div>
                    <div class="line-content">${this.applySyntaxHighlighting(line)}</div>
                </div>
            `;
        }).join('');
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
        
        // Animate the diff with insertions and deletions
        await this.animateFullDiff(prevLines, nextLines, container);
    }

    async animateFullDiff(prevLines, nextLines, container) {
        // Start by showing the previous version
        this.displayCode(prevLines.join('\n'), container);
        await this.delay(1000);
        
        // Two-sweep algorithm: delete all non-matching lines, then add all new lines
        let currentLines = [...prevLines];
        const nextSet = new Set(nextLines);
        
        // SWEEP 1: Delete all lines that don't appear in next (from end to start)
        // Process deletions one by one to maintain correct indexing
        const linesToDelete = [];
        for (let i = 0; i < currentLines.length; i++) {
            if (!nextSet.has(currentLines[i])) {
                linesToDelete.push(i);
            }
        }
        
        // Delete from end to start to maintain correct indices
        for (let idx = linesToDelete.length - 1; idx >= 0; idx--) {
            const lineIndex = linesToDelete[idx];
            const lineContent = currentLines[lineIndex];
            const tokens = this.tokenizeLine(lineContent);
            
            // Scroll to the line being deleted (adjust for already deleted lines)
            const displayLineNum = lineIndex + 1;
            this.scrollToLine(displayLineNum, container);
            
            // Highlight line for deletion
            const lineEl = container.querySelector(`[data-line="${displayLineNum}"]`);
            if (lineEl) {
                lineEl.classList.add('deleted');
            }
            
            // Delete tokens from end to start
            for (let tokenIdx = tokens.length - 1; tokenIdx >= 0; tokenIdx--) {
                // Rebuild the line without the last tokens
                const remainingTokens = tokens.slice(0, tokenIdx);
                currentLines[lineIndex] = remainingTokens.join('');
                
                // Update display with highlighted line
                const highlightedLines = [displayLineNum];
                this.displayCode(currentLines.join('\n'), container, highlightedLines);
                
                // Token delay (same as insertion)
                const delay = this.getTokenDelay(tokens[tokenIdx]);
                await this.delay(delay);
            }
            
            // Remove the empty line
            currentLines.splice(lineIndex, 1);
            this.displayCode(currentLines.join('\n'), container);
            await this.delay(25);
        }
        
        // Brief pause between sweeps
        await this.delay(300);
        
        // SWEEP 2: Add all lines from next in their correct positions
        // Build the final result line by line
        for (let targetIdx = 0; targetIdx < nextLines.length; targetIdx++) {
            const targetLine = nextLines[targetIdx];
            
            // Check if this line already exists at the correct position
            if (targetIdx < currentLines.length && currentLines[targetIdx] === targetLine) {
                // Line is already in the right place, skip
                continue;
            }
            
            // Find if this line exists elsewhere and remove it
            for (let i = targetIdx + 1; i < currentLines.length; i++) {
                if (currentLines[i] === targetLine) {
                    currentLines.splice(i, 1);
                    break;
                }
            }
            
            // Now insert the line at the correct position
            const tokens = this.tokenizeLine(targetLine);
            let currentLine = '';
            
            // Insert empty line at position
            currentLines.splice(targetIdx, 0, '');
            this.displayCode(currentLines.join('\n'), container);
            
            // Animate each token
            for (const token of tokens) {
                currentLine += token;
                currentLines[targetIdx] = currentLine;
                
                // Update display
                this.displayCode(currentLines.join('\n'), container);
                
                // Highlight the line being written
                const lineEl = container.querySelector(`[data-line="${targetIdx + 1}"]`);
                if (lineEl) {
                    lineEl.classList.add('added');
                    this.scrollToLine(targetIdx + 1, container);
                }
                
                // Token delay
                const delay = this.getTokenDelay(token);
                await this.delay(delay);
            }
            
            // Remove highlight after line is complete
            const lineEl = container.querySelector(`[data-line="${targetIdx + 1}"]`);
            if (lineEl) {
                setTimeout(() => {
                    lineEl.classList.remove('added');
                }, 400);
            }
            
            await this.delay(25);
        }
    }
    
    async writeNewContent(lines, container) {
        let currentLines = [];
        
        for (let i = 0; i < lines.length; i++) {
            const targetLine = lines[i];
            const tokens = this.tokenizeLine(targetLine);
            let currentLine = '';
            
            // Add empty line placeholder
            currentLines.push('');
            
            // Write tokens one by one
            for (const token of tokens) {
                currentLine += token;
                currentLines[i] = currentLine;
                
                // Update display
                this.displayCode(currentLines.join('\n'), container);
                
                // Highlight the line being written
                const lineEl = container.querySelector(`[data-line="${i + 1}"]`);
                if (lineEl) {
                    lineEl.classList.add('added');
                    this.scrollToLine(i + 1, container);
                }
                
                // Token delay
                const delay = this.getTokenDelay(token);
                await this.delay(delay);
            }
            
            // Remove highlight after line is complete
            const lineEl = container.querySelector(`[data-line="${i + 1}"]`);
            if (lineEl) {
                setTimeout(() => {
                    lineEl.classList.remove('added');
                }, 300);
            }
            
            // Pause between lines
            await this.delay(25);
        }
    }
    
    countCommonLines(prevLines, nextLines) {
        const nextSet = new Set(nextLines);
        return prevLines.filter(line => nextSet.has(line)).length;
    }
    
    
    async animateTokenByToken(prevLines, nextLines, container) {
        // Use LCS (Longest Common Subsequence) based approach for proper diff
        const diff = this.computeDiff(prevLines, nextLines);
        let currentLines = [...prevLines];
        let lineOffset = 0;
        
        for (const change of diff) {
            if (change.type === 'insert') {
                // Calculate actual insertion position with offset
                const insertPos = change.position + lineOffset;
                const targetLine = change.line;
                const tokens = this.tokenizeLine(targetLine);
                let currentLine = '';
                
                // Insert empty line at the position
                currentLines.splice(insertPos, 0, '');
                lineOffset++;
                
                // Animate each token
                for (const token of tokens) {
                    currentLine += token;
                    currentLines[insertPos] = currentLine;
                    
                    // Update display
                    this.displayCode(currentLines.join('\n'), container);
                    
                    // Highlight the line being written
                    const lineEl = container.querySelector(`[data-line="${insertPos + 1}"]`);
                    if (lineEl) {
                        lineEl.classList.add('added');
                        this.scrollToLine(insertPos + 1, container);
                    }
                    
                    // Token delay
                    const delay = this.getTokenDelay(token);
                    await this.delay(delay);
                }
                
                // Remove highlight after line is complete
                const lineEl = container.querySelector(`[data-line="${insertPos + 1}"]`);
                if (lineEl) {
                    setTimeout(() => {
                        lineEl.classList.remove('added');
                    }, 400);
                }
                
                // Pause between lines
                await this.delay(25);
            }
        }
    }
    
    computeDiff(prevLines, nextLines) {
        // Simple diff algorithm that finds insertions
        const changes = [];
        let prevIdx = 0;
        let nextIdx = 0;
        
        while (nextIdx < nextLines.length) {
            if (prevIdx < prevLines.length && prevLines[prevIdx] === nextLines[nextIdx]) {
                // Lines match, advance both
                prevIdx++;
                nextIdx++;
            } else {
                // Check if this line exists later in prev
                let found = false;
                for (let searchIdx = prevIdx; searchIdx < prevLines.length; searchIdx++) {
                    if (prevLines[searchIdx] === nextLines[nextIdx]) {
                        // Found it later - insert all new lines before it
                        found = true;
                        break;
                    }
                }
                
                if (!found) {
                    // This is a new line to insert
                    changes.push({
                        type: 'insert',
                        position: prevIdx,
                        line: nextLines[nextIdx]
                    });
                    nextIdx++;
                } else {
                    // Skip lines in prev that don't match current next
                    prevIdx++;
                }
            }
        }
        
        return changes;
    }
    
    tokenizeLine(line) {
        // Split line into meaningful tokens
        const tokens = [];
        const regex = /(\s+|[a-zA-Z_]\w*|[0-9]+|\S)/g;
        let match;
        
        while ((match = regex.exec(line)) !== null) {
            tokens.push(match[0]);
        }
        
        return tokens;
    }
    
    getTokenDelay(token) {
        // Variable delays for different token types (faster speed)
        if (token.match(/^\s+$/)) return 5;  // Whitespace
        if (token.match(/^[(){}\[\]]$/)) return 12;  // Brackets
        if (token.match(/^[a-zA-Z_]\w*$/)) return 10;  // Identifiers
        if (token.match(/^[0-9]+$/)) return 10;  // Numbers
        return 8;  // Default
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
            // Only scroll if the line is not already visible
            const rect = lineEl.getBoundingClientRect();
            const containerRect = container.getBoundingClientRect();
            
            if (rect.top < containerRect.top || rect.bottom > containerRect.bottom) {
                lineEl.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'center' 
                });
            }
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
