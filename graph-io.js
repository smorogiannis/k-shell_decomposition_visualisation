// Parse and export unweighted, undirected graphs for k-shell visualization.
// Supports edge-list (.txt), JSON.

function parseGraphFile(text, filename = '') {
    const trimmed = text.trim();
    if (!trimmed) throw new Error('File is empty.');

    const ext = (filename.split('.').pop() || '').toLowerCase();
    if (ext === 'json') return parseJsonGraph(trimmed);
    return parseEdgeList(trimmed);
}

function parseJsonGraph(text) {
    const data = JSON.parse(text);
    if (Array.isArray(data)) {
        return normalizeGraph([], data.map(row => {
            if (Array.isArray(row) && row.length >= 2) return { source: row[0], target: row[1] };
            throw new Error('JSON array must be pairs [[u,v], ...].');
        }));
    }
    if (data.edges) {
        const nodes = data.nodes || [];
        return normalizeGraph(nodes, data.edges);
    }
    throw new Error('JSON must be { nodes?, edges } or [[u,v], ...].');
}

function parseEdgeList(text) {
    const edges = [];
    const lines = text.split(/\r?\n/);

    for (let i = 0; i < lines.length; i++) {
        let line = lines[i].trim();
        if (!line || line.startsWith('#') || line.startsWith('%')) continue;

        // Skip common header rows
        if (/^(source|src|from|node1|u)\b/i.test(line) && /target|dst|to|node2|v/i.test(line)) continue;

        // "N nodes, M edges" metadata
        const meta = line.match(/^(\d+)\s+nodes?\b/i);
        if (meta) continue;

        const parts = line.split(/[\s,;\t|]+/).filter(Boolean);
        if (parts.length < 2) continue;

        let a = parts[0];
        let b = parts[1];
        // Optional weight column — ignored (unweighted only)
        edges.push({ source: a, target: b });
    }

    if (edges.length === 0) throw new Error('No edges found. Use one edge per line: "u v" or "u,v".');
    return normalizeGraph([], edges);
}

// Build canonical node list (1..N) and deduplicated undirected edges.
function normalizeGraph(rawNodes, rawEdges) {
    const labelToId = new Map();
    const nodes = [];
    let nextId = 1;

    function ensureNode(label) {
        const key = String(label).trim();
        if (!key) throw new Error('Empty node label.');
        if (!labelToId.has(key)) {
            const id = nextId++;
            labelToId.set(key, id);
            nodes.push({ id, label: key });
        }
        return labelToId.get(key);
    }

    rawNodes.forEach(n => {
        const label = n.label ?? n.id ?? n.name;
        if (label != null) ensureNode(label);
    });

    const edgeSet = new Set();
    const edges = [];

    rawEdges.forEach(e => {
        const src = e.source ?? e.src ?? e.from ?? e.u ?? e[0];
        const tgt = e.target ?? e.dst ?? e.to ?? e.v ?? e[1];
        if (src == null || tgt == null) return;

        const a = ensureNode(src);
        const b = ensureNode(tgt);
        if (a === b) return; // drop self-loops

        const lo = Math.min(a, b);
        const hi = Math.max(a, b);
        const key = `${lo}-${hi}`;
        if (edgeSet.has(key)) return;
        edgeSet.add(key);
        edges.push({ source: lo, target: hi });
    });

    if (nodes.length === 0 && edges.length > 0) {
        // ensureNode already populated nodes
    }
    if (nodes.length === 0) throw new Error('Graph has no nodes.');

    return { nodes, edges };
}

function layoutImportedNodes(nodes, width, height) {
    const n = nodes.length;
    const cx = width / 2;
    const cy = height / 2;
    const r = Math.min(width, height) * 0.35;

    nodes.forEach((node, i) => {
        const angle = (2 * Math.PI * i) / n;
        node.x = cx + r * Math.cos(angle);
        node.y = cy + r * Math.sin(angle);
        node.shell = null;
        node.degree = 0;
        node.removed = false;
    });
}

function exportEdgeList(nodes, edges) {
    const idToLabel = new Map(nodes.map(n => [n.id, n.label ?? n.id]));
    const lines = [
        '# k-shell lab export — unweighted undirected edge list',
        `# nodes: ${nodes.length}`,
        `# edges: ${edges.length}`,
        ''
    ];
    edges.forEach(e => {
        const a = idToLabel.get(e.source) ?? e.source;
        const b = idToLabel.get(e.target) ?? e.target;
        lines.push(`${a} ${b}`);
    });
    return lines.join('\n');
}

function exportJson(nodes, edges) {
    return JSON.stringify({
        directed: false,
        weighted: false,
        nodes: nodes.map(n => ({ id: n.id, label: n.label ?? String(n.id) })),
        edges: edges.map(e => ({ source: e.source, target: e.target }))
    }, null, 2);
}

function downloadText(content, filename, mime = 'text/plain') {
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        parseGraphFile,
        parseEdgeList,
        parseJsonGraph,
        normalizeGraph,
        layoutImportedNodes,
        exportEdgeList,
        exportJson,
        downloadText
    };
}
