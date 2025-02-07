# mypy: ignore-errors

from numba_scfg.core.datastructures.byte_flow import ByteFlow
from numba_scfg.rendering.rendering import render_flow


def scc(G):
    preorder = {}
    lowlink = {}
    scc_found = set()
    scc_queue = []
    out = []
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
                        out.append(scc)
                    else:
                        scc_queue.append(v)
    return out


def make_flow(func):
    return ByteFlow.from_bytecode(func)


def test_scc():
    f = make_flow(scc)
    f.scfg.restructure()


if __name__ == "__main__":
    render_flow(make_flow(scc))
