from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import networkx as nx

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return render_template('main.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_graph():
    data = request.json
    nodes = data.get('nodes', [])
    edges = data.get('edges', [])
    euler_path_user = data.get('eulerPath', []) 
    hamilton_path_user = data.get('hamiltonianPath', []) 

    original_edges = [e for e in edges if e.get('type') == 'original']
    user_added_edges = [e for e in edges if e.get('type') == 'user_added']

    def calc_bounds(nodes_list, edges_list):
        G = nx.MultiGraph()
        G_simple = nx.Graph()
        for n in nodes_list: 
            G.add_node(n['id'])
            G_simple.add_node(n['id'])
        for e in edges_list:
            G.add_edge(e['source'], e['target'])
            if not G_simple.has_edge(e['source'], e['target']):
                G_simple.add_edge(e['source'], e['target'], weight=1)
                
        comps = list(nx.connected_components(G))
        k = len(comps)
        O_total = sum(1 for v in G.nodes() if G.degree(v) % 2 != 0)
        k_0 = sum(1 for comp in comps if sum(1 for v in comp if G.degree(v) % 2 != 0) == 0)
        E_euler = O_total // 2 if k <= 1 else (O_total // 2) + k_0
        
        max_match = nx.max_weight_matching(G_simple, maxcardinality=True)
        E_match = (len(nodes_list) // 2) - len(max_match)
        return E_euler, E_match

    init_euler, init_match = calc_bounds(nodes, original_edges)
    curr_euler, curr_match = calc_bounds(nodes, edges)

    # ==========================================
    # 精確的聯合最少補邊計算 (Hamilton + Euler Parity)
    # ==========================================
    def get_combined_min_edges(nodes_list, edges_list):
        N = len(nodes_list)
        adj = {n['id']: set() for n in nodes_list}
        orig_deg = {n['id']: 0 for n in nodes_list}
        for e in edges_list:
            adj[e['source']].add(e['target'])
            adj[e['target']].add(e['source'])
            orig_deg[e['source']] += 1
            orig_deg[e['target']] += 1
            
        best_total = [999]
        best_hc_missing = [999]
        best_path = []
        steps = [0]
        
        start_node = list(adj.keys())[0]
        
        def dfs(curr, path, visited, current_missing):
            steps[0] += 1
            if steps[0] > 10000: return 
            if current_missing >= best_hc_missing[0]: return 
            
            if len(path) == N:
                is_missing_last = 1 if start_node not in adj[curr] else 0
                final_missing = current_missing + is_missing_last
                
                if final_missing < best_hc_missing[0]:
                    best_hc_missing[0] = final_missing
                    
                    new_deg = orig_deg.copy()
                    for i in range(N):
                        u, v = path[i], path[(i+1)%N]
                        if v not in adj[u]:
                            new_deg[u] += 1
                            new_deg[v] += 1
                            
                    odd_count = sum(1 for d in new_deg.values() if d % 2 != 0)
                    total_cost = final_missing + odd_count // 2
                    
                    if total_cost < best_total[0]:
                        best_total[0] = total_cost
                        best_path[:] = path[:] + [start_node]
                return
                
            neighbors, non_neighbors = [], []
            for nxt in [n['id'] for n in nodes_list]:
                if nxt not in visited:
                    if nxt in adj[curr]: neighbors.append(nxt)
                    else: non_neighbors.append(nxt)
            
            neighbors.sort(key=lambda x: len([v for v in adj[x] if v not in visited]))
            for nxt in neighbors:
                visited.add(nxt)
                path.append(nxt)
                dfs(nxt, path, visited, current_missing)
                path.pop()
                visited.remove(nxt)
                
            non_neighbors.sort(key=lambda x: len([v for v in adj[x] if v not in visited]))
            for nxt in non_neighbors:
                visited.add(nxt)
                path.append(nxt)
                dfs(nxt, path, visited, current_missing + 1)
                path.pop()
                visited.remove(nxt)

        dfs(start_node, [start_node], {start_node}, 0)
        if best_hc_missing[0] == 999: return 0, 0, []
        return best_total[0], best_hc_missing[0], best_path

    # 用於計算 Rollback 時的最佳路徑 (僅限 HC missing)
    def exact_hc_fixed(nodes_list, edges_list, fixed_path):
        N = len(nodes_list)
        adj = {n['id']: set() for n in nodes_list}
        for e in edges_list:
            adj[e['source']].add(e['target'])
            adj[e['target']].add(e['source'])
            
        best_missing = [999]
        best_path = []
        steps = [0]
        
        if not fixed_path: return 0, []
        
        start_node = fixed_path[0]
        curr_node = fixed_path[-1]
        visited = set(fixed_path)
        path = list(fixed_path)
        
        current_missing = 0
        for i in range(len(fixed_path)-1):
            if fixed_path[i+1] not in adj[fixed_path[i]]:
                current_missing += 1
                
        def dfs(curr, missing):
            steps[0] += 1
            if steps[0] > 10000: return
            if missing >= best_missing[0]: return 
            
            if len(path) == N:
                final_missing = missing + (0 if start_node in adj[curr] else 1)
                if final_missing < best_missing[0]:
                    best_missing[0] = final_missing
                    best_path[:] = path[:] + [start_node]
                return
                
            neighbors, non_neighbors = [], []
            for nxt in [n['id'] for n in nodes_list]:
                if nxt not in visited:
                    if nxt in adj[curr]: neighbors.append(nxt)
                    else: non_neighbors.append(nxt)
            
            neighbors.sort(key=lambda x: len([v for v in adj[x] if v not in visited]))
            for nxt in neighbors:
                visited.add(nxt)
                path.append(nxt)
                dfs(nxt, missing)
                path.pop()
                visited.remove(nxt)
                
            non_neighbors.sort(key=lambda x: len([v for v in adj[x] if v not in visited]))
            for nxt in non_neighbors:
                visited.add(nxt)
                path.append(nxt)
                dfs(nxt, missing + 1)
                path.pop()
                visited.remove(nxt)

        dfs(curr_node, current_missing)
        return best_missing[0], best_path

    init_total, init_ham, _ = get_combined_min_edges(nodes, original_edges)
    curr_total, curr_ham, optimal_curr_path = get_combined_min_edges(nodes, edges)
    
    estimation = {
        "init_euler": init_euler, "init_match": init_match, "init_ham": init_ham, 
        "init_total": init_total,
        "curr_euler": curr_euler, "curr_match": curr_match, "curr_ham": curr_ham, 
        "curr_total": curr_total,
        "user_added": len(user_added_edges)
    }

    # ==========================================
    # 打包缺少的漢米爾頓邊
    # ==========================================
    G_simple = nx.Graph()
    for edge in edges:
        u, v = edge['source'], edge['target']
        if not G_simple.has_edge(u, v): G_simple.add_edge(u, v)

    optimal_hc_missing_edges = []
    if optimal_curr_path:
        for i in range(len(optimal_curr_path)-1):
            u, v = optimal_curr_path[i], optimal_curr_path[i+1]
            if not G_simple.has_edge(u, v):
                optimal_hc_missing_edges.append({"source": u, "target": v})

    # ==========================================
    # 1. 尤拉迴圈演算法
    # ==========================================
    G_multi = nx.MultiGraph()
    for node in nodes: G_multi.add_node(node['id'])
    for i, edge in enumerate(edges): G_multi.add_edge(edge['source'], edge['target'], key=i)
    
    is_eulerian = nx.is_eulerian(G_multi)
    euler_result = { "is_eulerian": is_eulerian, "scenario": 0, "path": [], "edges_to_remove": [] }

    if is_eulerian:
        if len(euler_path_user) == 0:
            circuit = list(nx.eulerian_circuit(G_multi))
            euler_result["scenario"] = 1
            euler_result["path"] = [{"source": u, "target": v} for u, v in circuit]
        elif len(euler_path_user) == len(edges):
            euler_result["scenario"] = 3
        else:
            G_res = G_multi.copy()
            for step in euler_path_user:
                u, v, idx = step['from'], step['to'], step['edgeIndex']
                if G_res.has_edge(u, v, key=idx): G_res.remove_edge(u, v, key=idx)
                elif G_res.has_edge(v, u, key=idx): G_res.remove_edge(v, u, key=idx)

            curr_node = euler_path_user[-1]['to']
            start_node = euler_path_user[0]['from']

            def can_complete(H, u, v):
                if H.number_of_edges() == 0: return u == v
                for n in H.nodes():
                    deg = H.degree(n)
                    if n in (u, v) and u != v:
                        if deg % 2 == 0: return False
                    else:
                        if deg % 2 != 0: return False
                active_nodes = [n for n in H.nodes() if H.degree(n) > 0]
                if not active_nodes or u not in active_nodes: return False
                return nx.is_connected(H.subgraph(active_nodes))

            temp_path = euler_path_user.copy()
            rollback_count = 0
            valid_completion = False

            while temp_path:
                if can_complete(G_res, curr_node, start_node):
                    valid_completion = True
                    break
                step = temp_path.pop()
                rollback_count += 1
                G_res.add_edge(step['from'], step['to'], key=step['edgeIndex'])
                curr_node = temp_path[-1]['to'] if temp_path else start_node

            if not temp_path and not valid_completion: valid_completion = True

            euler_result["scenario"] = 2
            if rollback_count > 0: euler_result["edges_to_remove"] = euler_path_user[-rollback_count:]
            
            try:
                active_nodes = [n for n in G_res.nodes() if G_res.degree(n) > 0]
                if active_nodes:
                    G_active = G_res.subgraph(active_nodes).copy()
                    if curr_node in G_active:
                        path_edges = list(nx.eulerian_circuit(G_active, source=curr_node)) if curr_node == start_node else list(nx.eulerian_path(G_active, source=curr_node))
                        euler_result["path"] = [{"source": u, "target": v} for u, v in path_edges]
            except nx.NetworkXError: pass

    # ==========================================
    # 2. 完美匹配演算法
    # ==========================================
    for edge in edges:
        if edge.get('isMatched'):
            u, v = edge['source'], edge['target']
            if u > v: u, v = v, u 
            G_simple[u][v]['weight'] = 100

    matching = nx.max_weight_matching(G_simple, maxcardinality=True, weight='weight')
    has_perfect_matching = nx.is_perfect_matching(G_simple, matching)
    algo_pairs = set()
    for u, v in matching:
        if u > v: u, v = v, u
        algo_pairs.add((u, v))

    user_pairs = set()
    for edge in edges:
        if edge.get('isMatched'):
            u, v = edge['source'], edge['target']
            if u > v: u, v = v, u
            user_pairs.add((u, v))

    matching_result = { "has_perfect_matching": has_perfect_matching, "scenario": 0, "hint_edges": [], "edges_to_remove": [] }

    if has_perfect_matching:
        if len(user_pairs) == 0:
            matching_result["scenario"] = 1
            matching_result["hint_edges"] = [{"source": u, "target": v} for u, v in algo_pairs]
        elif len(user_pairs) == len(nodes) // 2 and user_pairs == algo_pairs:
            matching_result["scenario"] = 3
        else:
            remove_set = user_pairs - algo_pairs
            add_set = algo_pairs - user_pairs
            matching_result["scenario"] = 2
            matching_result["edges_to_remove"] = [{"source": u, "target": v} for u, v in remove_set]
            matching_result["hint_edges"] = [{"source": u, "target": v} for u, v in add_set]

    # ==========================================
    # 3. 漢米爾頓迴圈追蹤
    # ==========================================
    user_hc_nodes = []
    if hamilton_path_user:
        user_hc_nodes.append(hamilton_path_user[0]['from'])
        for p in hamilton_path_user: user_hc_nodes.append(p['to'])
        
    hc_result = { "has_hamiltonian": (curr_ham == 0), "scenario": 0, "path": [], "edges_to_remove": [], "hint_edges": [] }

    has_dup = len(set(user_hc_nodes)) < len(user_hc_nodes)
    if has_dup and len(user_hc_nodes) <= len(nodes):
        dup_idx = 0
        seen = set()
        for i, n in enumerate(user_hc_nodes):
            if n in seen: dup_idx = i; break
            seen.add(n)
        rb = len(user_hc_nodes) - dup_idx
        hc_result["scenario"] = 2
        hc_result["edges_to_remove"] = hamilton_path_user[-rb:]
        
        valid_nodes = user_hc_nodes[:-rb]
        _, best_p = exact_hc_fixed(nodes, edges, valid_nodes)
        if not best_p: best_p = optimal_curr_path
        
        hc_result["path"] = [{"source": best_p[i], "target": best_p[i+1]} for i in range(len(valid_nodes)-1, len(best_p)-1)]
        for i in range(len(best_p)-1):
            u, v = best_p[i], best_p[i+1]
            if not G_simple.has_edge(u, v):
                hc_result["hint_edges"].append({"source": u, "target": v})
                
    elif user_hc_nodes:
        user_path_missing, best_user_p = exact_hc_fixed(nodes, edges, user_hc_nodes)
        
        # 【核心修正】判斷是否逼著玩家加超過 curr_ham 的邊，若是，強制倒退！
        if user_path_missing > curr_ham:
            best_rb = len(user_hc_nodes)
            best_p = optimal_curr_path
            for rb in range(1, len(user_hc_nodes)):
                test_nodes = user_hc_nodes[:-rb]
                m, p = exact_hc_fixed(nodes, edges, test_nodes)
                if m == curr_ham:
                    best_rb = rb
                    best_p = p
                    break
                    
            hc_result["scenario"] = 2
            hc_result["edges_to_remove"] = hamilton_path_user[-best_rb:]
            valid_nodes = user_hc_nodes[:-best_rb]
            hc_result["path"] = [{"source": best_p[i], "target": best_p[i+1]} for i in range(max(0, len(valid_nodes)-1), len(best_p)-1)]
            
            for i in range(len(best_p)-1):
                u, v = best_p[i], best_p[i+1]
                if not G_simple.has_edge(u, v):
                    hc_result["hint_edges"].append({"source": u, "target": v})
                    
        else:
            if len(hamilton_path_user) == len(nodes) and hamilton_path_user[-1]['to'] == hamilton_path_user[0]['from']:
                hc_result["scenario"] = 3
            else:
                hc_result["scenario"] = 2
                hc_result["path"] = [{"source": best_user_p[i], "target": best_user_p[i+1]} for i in range(max(0, len(user_hc_nodes)-1), len(best_user_p)-1)]
                for i in range(len(best_user_p)-1):
                    u, v = best_user_p[i], best_user_p[i+1]
                    if not G_simple.has_edge(u, v):
                        hc_result["hint_edges"].append({"source": u, "target": v})
    else:
        hc_result["scenario"] = 1
        hc_result["path"] = [{"source": optimal_curr_path[i], "target": optimal_curr_path[i+1]} for i in range(len(optimal_curr_path)-1)]
        hc_result["hint_edges"] = optimal_hc_missing_edges

    return jsonify({
        "euler_info": euler_result,
        "matching_info": matching_result,
        "hamilton_info": hc_result,
        "estimation": estimation,
        "optimal_hc_missing_edges": optimal_hc_missing_edges
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)