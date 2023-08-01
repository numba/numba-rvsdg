"""
SCC from https://github.com/networkx/networkx/blob/41a760273eed666e6f966cf14ea524dec56d678b/networkx/algorithms/components/strongly_connected.py#L16 # noqa

LICENSE: https://github.com/networkx/networkx/blob/main/LICENSE.txt

"""

# Ignore all mypy errors since this file has been vendored.
# mypy: ignore-errors


def scc(G):
    preorder = {}
    lowlink = {}
    scc_found = set()
    scc_queue = []
    i = 0  # Preorder counter
    for source in G:
        if source not in scc_found:
            queue = [source]
            while queue:
                v = queue[-1]
                if v not in preorder:
                    i = i + 1
                    preorder[v] = i
                done = True
                for w in G[v]:
                    if w not in preorder:
                        queue.append(w)
                        done = False
                        break
                if done:
                    lowlink[v] = preorder[v]
                    for w in G[v]:
                        if w not in scc_found:
                            if preorder[w] > preorder[v]:
                                lowlink[v] = min([lowlink[v], lowlink[w]])
                            else:
                                lowlink[v] = min([lowlink[v], preorder[w]])
                    queue.pop()
                    if lowlink[v] == preorder[v]:
                        scc = {v}
                        while (
                            scc_queue and preorder[scc_queue[-1]] > preorder[v]
                        ):
                            k = scc_queue.pop()
                            scc.add(k)
                        scc_found.update(scc)
                        yield scc
                    else:
                        scc_queue.append(v)


def sccr(G):
    def visit(v, cnt):
        root[v] = cnt
        visited[v] = cnt
        cnt += 1
        stack.append(v)
        for w in G[v]:
            if w not in visited:
                yield from visit(w, cnt)
            if w not in component:
                root[v] = min(root[v], root[w])
        if root[v] == visited[v]:
            component[v] = root[v]
            tmpc = {v}  # hold nodes in this component
            while stack[-1] != v:
                w = stack.pop()
                component[w] = root[v]
                tmpc.add(w)
            stack.remove(v)
            yield tmpc

    visited = {}
    component = {}
    root = {}
    cnt = 0
    stack = []
    for source in G:
        if source not in visited:
            yield from visit(source, cnt)
