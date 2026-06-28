const DIFFUSION_THRESHOLD = 2;

function edgeKey(a, b) {
    return a < b ? `${a}-${b}` : `${b}-${a}`;
}

function neighborsOf(nodeId, edges) {
    const out = [];
    edges.forEach(e => {
        if (e.removed) return;
        if (e.source === nodeId) out.push(e.target);
        else if (e.target === nodeId) out.push(e.source);
    });
    return out;
}

// Standard k-shell decomposition.
// At each level k, repeatedly prune nodes whose current degree <= k until none
// remain, assign them shell k, then move to k+1. Each shell may take several
// pruning rounds; we record the round (the within-shell "stage") per node and
// the max round per shell, both consumed by the gravity model below.
function standardKShellDecomposition(nodes, edges) {
    const workingNodes = nodes.map(n => ({...n, shell: null, removed: false}));
    const workingEdges = edges.map(e => ({...e, removed: false}));

    const steps = [];
    const stageByNode = {};
    const globalIterByNode = {};
    const qK = {};
    let totalGlobalIters = 0;

    const recomputeDegrees = () => {
        workingNodes.forEach(n => {
            if (n.removed) return;
            n.degree = 0;
            workingEdges.forEach(e => {
                if (e.removed) return;
                if (e.source !== n.id && e.target !== n.id) return;
                const otherId = e.source === n.id ? e.target : e.source;
                const other = workingNodes.find(m => m.id === otherId);
                if (other && !other.removed) n.degree++;
            });
        });
    };

    let k = 0;
    while (workingNodes.some(n => !n.removed)) {
        let stage = 0;
        let removedThisStage;
        do {
            removedThisStage = false;
            recomputeDegrees();
            const toRemove = workingNodes.filter(n => !n.removed && n.degree <= k);
            if (toRemove.length > 0) {
                stage++;
                totalGlobalIters++;
                removedThisStage = true;
                toRemove.forEach(node => {
                    node.removed = true;
                    node.shell = k;
                    stageByNode[node.id] = stage;
                    globalIterByNode[node.id] = totalGlobalIters;
                    steps.push({
                        type: 'remove_node',
                        nodeId: node.id,
                        shell: k,
                        stage: stage,
                        reason: `Degree ${node.degree} ≤ k=${k}`
                    });
                });
            }
        } while (removedThisStage);
        if (stage > 0) qK[k] = stage;
        k++;
    }

    return {
        algorithm: 'Standard K-Shell',
        steps,
        maxShell: k - 1,
        shellAssignments: workingNodes.reduce((acc, n) => {
            acc[n.id] = n.shell;
            return acc;
        }, {}),
        stageByNode,
        globalIterByNode,
        totalGlobalIters,
        qK
    };
}

function buildAdjacency(nodes, edges) {
    const adjacency = {};
    nodes.forEach(n => { adjacency[n.id] = []; });
    edges.forEach(e => {
        if (e.removed) return;
        adjacency[e.source].push(e.target);
        adjacency[e.target].push(e.source);
    });
    return adjacency;
}

function localClusteringCoefficient(nodeId, adjacency) {
    const neighbors = adjacency[nodeId] || [];
    const ki = neighbors.length;
    if (ki <= 1) return 0;
    let triangles = 0;
    for (let a = 0; a < neighbors.length; a++) {
        for (let b = a + 1; b < neighbors.length; b++) {
            const na = neighbors[a];
            const nb = neighbors[b];
            if ((adjacency[na] || []).includes(nb)) triangles++;
        }
    }
    return (2 * triangles) / (ki * (ki - 1));
}

function countSecondOrderNeighbors(nodeId, adjacency) {
    const first = new Set(adjacency[nodeId] || []);
    const second = new Set();
    first.forEach(j => {
        (adjacency[j] || []).forEach(k => {
            if (k !== nodeId && !first.has(k)) second.add(k);
        });
    });
    return second.size;
}

function entropyTerm(p) {
    return p > 0 ? -p * Math.log(p) : 0;
}

function attributeEntropyWeight(values) {
    const n = values.length;
    if (n <= 1) return 0;
    const sum = values.reduce((a, b) => a + b, 0);
    if (sum === 0) return 0;
    const lnN = Math.log(n);
    if (!lnN) return 0;
    const norm = values.map(v => v / sum);
    return norm.reduce((acc, p) => acc + entropyTerm(p), 0) / lnN;
}

// Node Propagation Entropy
function nodePropagationEntropyKShell(nodes, edges) {
    const standardResults = standardKShellDecomposition(nodes, edges);
    const adjacency = buildAdjacency(nodes, edges);
    const propagationEntropy = {};

    const cnByNode = {};
    let sumCn = 0;
    nodes.forEach(node => {
        const id = node.id;
        const firstOrder = (adjacency[id] || []).length;
        const secondOrder = countSecondOrderNeighbors(id, adjacency);
        const clustering = localClusteringCoefficient(id, adjacency);
        const cn = (firstOrder + secondOrder) / (1 + clustering);
        cnByNode[id] = cn;
        sumCn += cn;
    });

    const iByNode = {};
    nodes.forEach(node => {
        iByNode[node.id] = sumCn > 0 ? cnByNode[node.id] / sumCn : 0;
    });

    nodes.forEach(node => {
        const id = node.id;
        let pe = 0;
        (adjacency[id] || []).forEach(j => {
            pe += entropyTerm(iByNode[j]);
        });
        propagationEntropy[id] = {
            clustering: localClusteringCoefficient(id, adjacency),
            firstOrder: (adjacency[id] || []).length,
            secondOrder: countSecondOrderNeighbors(id, adjacency),
            cn: cnByNode[id],
            pe
        };
    });

    return {
        algorithm: 'Node Propagation Entropy',
        steps: standardResults.steps,
        maxShell: standardResults.maxShell,
        shellAssignments: standardResults.shellAssignments,
        propagationEntropy
    };
}

function sigmoidPositionIndex(iter) {
    if (!iter) return 0.75;
    return 0.75 / (1 + Math.exp(-Math.sqrt(iter)));
}

// Information Entropy k-shell
function informationEntropyKShell(nodes, edges) {
    const standardResults = standardKShellDecomposition(nodes, edges);
    const { shellAssignments, globalIterByNode, totalGlobalIters } = standardResults;
    const adjacency = buildAdjacency(nodes, edges);
    const degreeById = {};
    nodes.forEach(node => { degreeById[node.id] = node.degree; });

    const positionIndex = {};
    nodes.forEach(node => {
        const iter = globalIterByNode[node.id] || 1;
        positionIndex[node.id] = sigmoidPositionIndex(iter);
    });

    const pnpByNode = {};
    nodes.forEach(node => {
        const id = node.id;
        const ks = shellAssignments[id] ?? 0;
        let neighborPos = 0;
        (adjacency[id] || []).forEach(j => { neighborPos += positionIndex[j]; });
        pnpByNode[id] = ks + neighborPos;
    });

    const pnnByNode = {};
    nodes.forEach(node => {
        const id = node.id;
        let sum = 0;
        (adjacency[id] || []).forEach(j => {
            (adjacency[j] || []).forEach(l => {
                sum += degreeById[l] || 0;
            });
        });
        pnnByNode[id] = sum;
    });

    const pnpValues = nodes.map(n => pnpByNode[n.id]);
    const pnnValues = nodes.map(n => pnnByNode[n.id]);
    const H1 = attributeEntropyWeight(pnpValues);
    const H2 = attributeEntropyWeight(pnnValues);
    const weightDenom = (2 - H1 - H2) || 1;
    const w1 = (1 - H1) / weightDenom;
    const w2 = (1 - H2) / weightDenom;

    const entropyKShell = {};
    nodes.forEach(node => {
        const id = node.id;
        entropyKShell[id] = {
            iter: globalIterByNode[id] || 1,
            positionIndex: positionIndex[id],
            pnp: pnpByNode[id],
            pnn: pnnByNode[id],
            pn: w1 * pnpByNode[id] + w2 * pnnByNode[id]
        };
    });

    return {
        algorithm: 'Information Entropy K-Shell',
        steps: standardResults.steps,
        maxShell: standardResults.maxShell,
        shellAssignments,
        entropyWeights: { w1, w2 },
        entropyKShell
    };
}

// Redundant Link Removal Algorithm
//   D_ij = (n_{i->j} + n_{j->i}) / 2
// where n_{i->j} counts neighbors of j that lie outside i's closed neighborhood
// and symmetrically. Redundant if D_ij < D_thr (from paper: D_thr = 2);
// then k-shell runs on the residual graph G′ with the same node set.
function calculateDiffusionImportance(edge, edges) {
    const sourceNeighbors = neighborsOf(edge.source, edges).filter(id => id !== edge.target);
    const targetNeighbors = neighborsOf(edge.target, edges).filter(id => id !== edge.source);

    const n_i_to_j = targetNeighbors.filter(id => !sourceNeighbors.includes(id)).length;
    const n_j_to_i = sourceNeighbors.filter(id => !targetNeighbors.includes(id)).length;

    return (n_i_to_j + n_j_to_i) / 2;
}

function redundantLinkRemovalKShell(nodes, edges) {
    const activeEdges = edges.filter(e => !e.removed);
    const originalKShell = standardKShellDecomposition(nodes, activeEdges);

    const edgeImportance = {};
    activeEdges.forEach(edge => {
        edgeImportance[edgeKey(edge.source, edge.target)] =
            calculateDiffusionImportance(edge, activeEdges);
    });

    const residualEdges = activeEdges.filter(edge => {
        return edgeImportance[edgeKey(edge.source, edge.target)] >= DIFFUSION_THRESHOLD;
    });
    const removedEdges = activeEdges.filter(edge => {
        return edgeImportance[edgeKey(edge.source, edge.target)] < DIFFUSION_THRESHOLD;
    });

    const results = standardKShellDecomposition(nodes, residualEdges);

    // Prepend remove_edge events so the animation can show redundant links
    // being pruned before any node removals.
    const edgeSteps = removedEdges.map(edge => {
        const key = edgeKey(edge.source, edge.target);
        const importance = edgeImportance[key];
        return {
            type: 'remove_edge',
            source: edge.source,
            target: edge.target,
            importance,
            reason: `D_ij = ${importance.toFixed(2)} <= ${DIFFUSION_THRESHOLD}`
        };
    });
    results.steps = edgeSteps.concat(results.steps);

    results.algorithm = 'Redundant Link Removal';
    results.removedEdges = removedEdges;
    results.residualEdges = residualEdges;
    results.edgeImportance = edgeImportance;
    results.originalShellAssignments = originalKShell.shellAssignments;
    results.originalMaxShell = originalKShell.maxShell;
    return results;
}

// BFS-based (Breadth-First Search) unweighted shortest paths from every node to every other.
// Returns dist[u][v] = number of edges on the shortest u->v path, undefined if
// there is no path.
function allPairsShortestPaths(nodes, edges) {
    const adjacency = {};
    nodes.forEach(n => { adjacency[n.id] = []; });
    edges.forEach(e => {
        if (adjacency[e.source]) adjacency[e.source].push(e.target);
        if (adjacency[e.target]) adjacency[e.target].push(e.source);
    });

    const dist = {};
    nodes.forEach(start => {
        dist[start.id] = { [start.id]: 0 };
        const queue = [start.id];
        while (queue.length > 0) {
            const u = queue.shift();
            adjacency[u].forEach(v => {
                if (dist[start.id][v] === undefined) {
                    dist[start.id][v] = dist[start.id][u] + 1;
                    queue.push(v);
                }
            });
        }
    });
    return dist;
}

// Improved Gravity Model Algorithm
function averageShortestPathDistance(nodes, distances) {
    let sum = 0;
    let count = 0;
    for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
            const d = distances[nodes[i].id] && distances[nodes[i].id][nodes[j].id];
            if (d) {
                sum += d;
                count++;
            }
        }
    }
    return count ? sum / count : 1;
}

function dkGravityModelKShell(nodes, edges) {
    const standardResults = standardKShellDecomposition(nodes, edges);
    const { stageByNode, qK, shellAssignments } = standardResults;

    const qValues = Object.values(qK);
    const qMax = qValues.length ? Math.max(...qValues) : 1;
    const denom = qMax + 1;

    const improvedKShell = {};
    nodes.forEach(node => {
        const k = shellAssignments[node.id];
        if (k === null || k === undefined) return;
        const p = stageByNode[node.id] || 1;
        const ksStar = k + p / denom;
        const dk = node.degree + ksStar;
        improvedKShell[node.id] = {
            originalShell: k,
            stage: p,
            qK: qK[k] || 1,
            qMax: qMax,
            improvedShell: ksStar,
            dkValue: dk,
            mass: dk
        };
    });

    const distances = allPairsShortestPaths(nodes, edges);
    const avgDist = averageShortestPathDistance(nodes, distances);
    const truncationRadius = Math.max(2, Math.round(avgDist));

    nodes.forEach(node => {
        const info = improvedKShell[node.id];
        if (!info) return;
        let dkgm = 0;
        nodes.forEach(other => {
            if (other.id === node.id) return;
            const otherInfo = improvedKShell[other.id];
            if (!otherInfo) return;
            const d = distances[node.id] && distances[node.id][other.id];
            if (!d || d > truncationRadius) return;
            dkgm += (info.dkValue * otherInfo.dkValue) / (d * d);
        });
        info.dkgm = dkgm;
        info.gravity = dkgm;
    });

    return {
        algorithm: 'Improved Gravity Model',
        steps: standardResults.steps,
        maxShell: standardResults.maxShell,
        shellAssignments,
        improvedKShell,
        truncationRadius,
        averageDistance: avgDist
    };
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        DIFFUSION_THRESHOLD,
        edgeKey,
        standardKShellDecomposition,
        calculateDiffusionImportance,
        redundantLinkRemovalKShell,
        allPairsShortestPaths,
        averageShortestPathDistance,
        dkGravityModelKShell,
        buildAdjacency,
        localClusteringCoefficient,
        countSecondOrderNeighbors,
        nodePropagationEntropyKShell,
        informationEntropyKShell
    };
}
