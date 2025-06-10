---
date : '2025-04-26T14:44:25+08:00'
draft : false
title : 'Bellman-Ford算法原理及Python实现'
image: index.png
categories:
  - 学习
tags:
  - 算法
  - 图论
---
## Bellman-Ford算法原理及Python实现

---

## 算法简介

Bellman-Ford算法主要用于求解有向图的**单源最短路径**问题，与迪杰斯特拉算法不同，他可以处理**带有负权值的图**，并且可以检测图中是否有负权环（负权环指的是从源点到源点的一个环，并且环上权重和为负数）。

---

## 算法原理

​	它的基本思想是松弛（Relaxation）操作。松弛是指对于每一条边（u，v），如果从源点到顶点 u 的最短路径距离已知，并且从源点到顶点 v 的距离可以通过经过顶点 u 的路径来更新为一个更小的值，那么就更新顶点 v 的当前最短路径距离。

​	算法重复进行松弛操作，对于图中的每一条边都进行检查，尝试更新顶点的最短路径估计值。这个过程需要进行 |V| - 1 次（|V| 是图中顶点的数量），因为在最坏情况下，一个顶点的最短路径可能需要经过所有其他顶点。

​	因此算法的时间复杂度为 O（|V|・|E|），其中 |V| 是顶点数，|E| 是边数。

---

## 算法流程

### 步骤1：初始化

将源点的最短路径距离设为 0，其他所有顶点的最短路径距离设为无穷大。

### 步骤2：反复进行松弛操作

对于每一条边（u，v），在 |V| - 1 次迭代中，如果当前顶点 u 的最短路径距离加上边（u，v）的权重小于顶点 v 当前记录的最短路径距离，则更新顶点 v 的最短路径距离。

### 步骤3：检测是否存在负权回路

在完成 |V| - 1 次松弛操作后，再对所有边进行一次检查。如果还能松弛，说明图中存在从源点可达的负权回路，此时最短路径不存在（因为可以无限绕负权回路来降低路径权重）。

---

## Python实现

```python
class Edge:
    def __init__(self, src, dest, weight):
        self.src = src
        self.dest = dest
        self.weight = weight

def bellman_ford(vertices, edges, src):
    # 初始化距离数组，距离源点的距离为无穷大
    dist = [float('inf')] * (vertices + 1)
    dist[src] = 0  # 源点到自身的距离为 0

    # 进行 vertices-1 次松弛操作
    for _ in range(vertices - 1):
        updated = False
        for edge in edges:
            u = edge.src
            v = edge.dest
            weight = edge.weight
            if dist[u] != float('inf') and dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                updated = True
        if not updated:
            break  # 如果没有边可以松弛，提前退出

    # 检测是否存在负权回路
    has_negative_cycle = False
    for edge in edges:
        u = edge.src
        v = edge.dest
        weight = edge.weight
        if dist[u] != float('inf') and dist[v] > dist[u] + weight:
            has_negative_cycle = True
            break

    return dist, has_negative_cycle
```

---

## 输入与输出

输入为图的邻接矩阵，其中 `graph[i][j]` 表示从节点 `i` 到节点 `j` 的边的权重。如果节点之间没有直接的边，则用一个足够大的值（如 `max`）表示不可达。

输出为 `path` 数组，其中 `path[i]` 表示从起点到节点 `i` 的最短路径中，到达节点 `i` 的前一个节点的索引。如果节点不可达，则 `path[i]` 为 `-1`

若想通过path数组得到从起点到某个节点k的路径，可由如下代码实现：

```python
def get_path(path, start_index, target_index):
    if path[target_index] == -1:
        return "没有路径可达"
    path_list = []
    while target_index != start_index:
        path_list.append(target_index)
        target_index = path[target_index]
    path_list.append(start_index)
    path_list.reverse()
    return path_list
```

---

## 代码测试

主程序测试代码为：

```python
if __name__ == "__main__":
    # 图1的顶点数为 5
    vertices1 = 5
    # 图1的边集合
    edges1 = [
        Edge(1, 2, 4),
        Edge(1, 3, 2),
        Edge(2, 3, 5),
        Edge(2, 4, 3),
        Edge(3, 2, -3),
        Edge(3, 5, 7),
        Edge(4, 5, 1),
        Edge(5, 1, 8)
    ]
    # 源点为 1
    src1 = 1
    distances1, has_negative_cycle1 = bellman_ford(vertices1, edges1, src1)
    print("图1示例：无负权环图")
    if has_negative_cycle1:
        print("图中存在负权回路")
    else:
        print("源点为", src1, "的最短距离为:")
        for i in range(1, vertices1 + 1):
            print("到顶点", i, "的距离为:", distances1[i])
	#-------------------------------------分割线-----------------------------------------------
    # 图2的顶点数为 3
    vertices2 = 3
    edges2 = [
        Edge(1, 2, 1),
        Edge(2, 3, 2),
        Edge(3, 1, -4)  # 这条边形成一个负权环
    ]
    src2 = 1
    distances2, has_negative_cycle2 = bellman_ford(vertices2, edges2, src2)
    print("图2示例：有负权环图")
    if has_negative_cycle2:
        print("图中存在负权回路")
    else:
        print("源点为", src2, "的最短距离为:")
        for i in range(1, vertices2 + 1):
            print("到顶点", i, "的距离为:", distances2[i])
```

运行结果为：

```
E:\BLOG_article\Bellman-Ford\.venv\Scripts\python.exe E:\BLOG_article\Bellman-Ford\Bellman-Ford.py 
图1示例：无负权环图
源点为 1 的最短距离为:
到顶点 1 的距离为: 0
到顶点 2 的距离为: -1
到顶点 3 的距离为: 2
到顶点 4 的距离为: 2
到顶点 5 的距离为: 3

图2示例：有负权环图
图中存在负权回路
```

## 完整代码

```python
class Edge:
    def __init__(self, src, dest, weight):
        self.src = src
        self.dest = dest
        self.weight = weight

def bellman_ford(vertices, edges, src):
    # 初始化距离数组，距离源点的距离为无穷大
    dist = [float('inf')] * (vertices + 1)
    dist[src] = 0  # 源点到自身的距离为 0

    # 进行 vertices-1 次松弛操作
    for _ in range(vertices - 1):
        updated = False
        for edge in edges:
            u = edge.src
            v = edge.dest
            weight = edge.weight
            if dist[u] != float('inf') and dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                updated = True
        if not updated:
            break  # 如果没有边可以松弛，提前退出

    # 检测是否存在负权回路
    has_negative_cycle = False
    for edge in edges:
        u = edge.src
        v = edge.dest
        weight = edge.weight
        if dist[u] != float('inf') and dist[v] > dist[u] + weight:
            has_negative_cycle = True
            break

    return dist, has_negative_cycle

if __name__ == "__main__":
    # 图1的顶点数为 5
    vertices1 = 5
    # 图1的边集合
    edges1 = [
        Edge(1, 2, 4),
        Edge(1, 3, 2),
        Edge(2, 3, 5),
        Edge(2, 4, 3),
        Edge(3, 2, -3),
        Edge(3, 5, 7),
        Edge(4, 5, 1),
        Edge(5, 1, 8)
    ]
    # 源点为 1
    src1 = 1
    distances1, has_negative_cycle1 = bellman_ford(vertices1, edges1, src1)
    print("图1示例：无负权环图")
    if has_negative_cycle1:
        print("图中存在负权回路")
    else:
        print("源点为", src1, "的最短距离为:")
        for i in range(1, vertices1 + 1):
            print("到顶点", i, "的距离为:", distances1[i])
	#-------------------------------------分割线-----------------------------------------------
    # 图2的顶点数为 3
    vertices2 = 3
    edges2 = [
        Edge(1, 2, 1),
        Edge(2, 3, 2),
        Edge(3, 1, -4)  # 这条边形成一个负权环
    ]
    src2 = 1
    distances2, has_negative_cycle2 = bellman_ford(vertices2, edges2, src2)
    print("图2示例：有负权环图")
    if has_negative_cycle2:
        print("图中存在负权回路")
    else:
        print("源点为", src2, "的最短距离为:")
        for i in range(1, vertices2 + 1):
            print("到顶点", i, "的距离为:", distances2[i])
```

