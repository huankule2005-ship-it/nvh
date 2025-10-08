import heapq
import matplotlib.pyplot as plt
import numpy as np

# ---------- CẤU HÌNH MÔI TRƯỜNG ----------
W, H = 20, 14  # kích thước bản đồ (rộng, cao)
start = (1, 1)
goal = (18, 12)
max_time = 60  # giới hạn số bước thời gian

# Chướng ngại vật tĩnh (tường, bàn)
static_obstacles = {
    (5, y) for y in range(1, 6)
} | {
    (10, y) for y in range(4, 13)
} | {
    (14, y) for y in range(0, 6)
} | {
    (7, 8), (8, 8), (9, 8)
}

# Vùng cấm (forbidden zone)
forbidden_zones = {(3, 10), (4, 10), (3,11), (4,11), (12,2), (13,2)}

# ---------- VẬT THỂ DI ĐỘNG ----------
def moving_obj_1_pos(t):
    # Di chuyển qua lại trên hàng y=6 giữa x=2..9
    period = 14
    r = t % period
    if r < 8:
        x = 2 + r
    else:
        x = 2 + (period - r)
    return (x, 6)

def moving_obj_2_pos(t):
    # Đường đi hình vòng lặp
    seq = [(16,5),(16,6),(15,7),(14,7),(13,7),(12,7),
           (11,6),(11,5),(11,4),(12,4),(13,4),(14,4),
           (15,4),(16,4)]
    return seq[t % len(seq)]

moving_objects = [moving_obj_1_pos, moving_obj_2_pos]

def occupied_by_moving(x, y, t):
    """Kiểm tra tại thời điểm t có vật thể di động ở (x, y) không"""
    for f in moving_objects:
        if f(t) == (x, y):
            return True
    return False

# ---------- CÀI ĐẶT UCS ----------
actions = [(1,0), (-1,0), (0,1), (0,-1), (0,0)]  # 4 hướng + đứng yên

def in_bounds(x,y):
    return 0 <= x < W and 0 <= y < H

def is_blocked(x,y):
    return (x,y) in static_obstacles or (x,y) in forbidden_zones

def collision(robot_from, robot_to, t):
    """Tránh va chạm trực tiếp hoặc đổi chỗ với vật di động"""
    if occupied_by_moving(robot_to[0], robot_to[1], t+1):
        return True
    for f in moving_objects:
        pos_t = f(t)
        pos_t1 = f(t+1)
        if pos_t == robot_to and pos_t1 == robot_from:
            return True
    return False

def move_cost(from_cell, to_cell):
    return 1.0  # chi phí di chuyển mỗi bước là 1

def uniform_cost_search(start, goal, max_time):
    sx, sy = start
    gx, gy = goal
    pq = []
    heapq.heappush(pq, (0.0, sx, sy, 0, None))
    best_cost = {(sx, sy, 0): 0.0}
    parents = {}

    while pq:
        cost, x, y, t, _ = heapq.heappop(pq)
        if (x, y) == (gx, gy):
            # Tái tạo đường đi
            path = []
            state = (x, y, t)
            while state is not None:
                path.append(state)
                state = parents.get(state)
            path.reverse()
            return path, cost

        if t >= max_time:
            continue

        for dx, dy in actions:
            nx, ny = x + dx, y + dy
            nt = t + 1
            if not in_bounds(nx, ny): continue
            if is_blocked(nx, ny): continue
            if occupied_by_moving(nx, ny, nt): continue
            if collision((x, y), (nx, ny), t): continue

            ncost = cost + move_cost((x, y), (nx, ny))
            state = (nx, ny, nt)
            if best_cost.get(state, float('inf')) > ncost:
                best_cost[state] = ncost
                parents[state] = (x, y, t)
                heapq.heappush(pq, (ncost, nx, ny, nt, (x, y, t)))

    return None, float('inf')

# ---------- CHẠY THUẬT TOÁN ----------
path, total_cost = uniform_cost_search(start, goal, max_time)
if path is None:
    print("❌ Không tìm thấy đường đi khả thi.")
else:
    print(f"✅ Tìm được đường đi với chi phí {total_cost:.2f}, số bước: {len(path)}")

# ---------- VẼ MINH HỌA ----------
fig, ax = plt.subplots(figsize=(10,7))
ax.set_aspect('equal')
ax.set_xlim(-0.5, W-0.5)
ax.set_ylim(-0.5, H-0.5)
ax.invert_yaxis()
ax.set_xticks(range(W))
ax.set_yticks(range(H))
ax.grid(True, linewidth=0.6)

# Vẽ vật cản tĩnh
for (ox, oy) in static_obstacles:
    rect = plt.Rectangle((ox-0.5, oy-0.5), 1,1, color='gray')
    ax.add_patch(rect)

# Vẽ vùng cấm
for (fx, fy) in forbidden_zones:
    rect = plt.Rectangle((fx-0.5, fy-0.5), 1,1, hatch='///', fill=False, edgecolor='red')
    ax.add_patch(rect)

# Vẽ quỹ đạo vật thể di động (30 bước đầu)
for f in moving_objects:
    traj = [f(t) for t in range(30)]
    xs = [p[0] for p in traj]
    ys = [p[1] for p in traj]
    ax.plot(xs, ys, '--o', markersize=4)

# Vẽ điểm bắt đầu và đích
ax.text(start[0], start[1], 'S', fontsize=12, fontweight='bold', ha='center', va='center', color='green')
ax.text(goal[0], goal[1], 'G', fontsize=12, fontweight='bold', ha='center', va='center', color='red')

# Vẽ đường đi
if path:
    path_xy = [(x, y) for (x, y, t) in path]
    xs = [p[0] for p in path_xy]
    ys = [p[1] for p in path_xy]
    ax.plot(xs, ys, linewidth=2, color='blue')
    for (x, y, t) in path[:: max(1, len(path)//15)]:
        ax.text(x, y, str(t), fontsize=8, ha='center', va='center')

ax.set_title("Đường đi robot (UCS) tránh vật cản, vật thể di động và vùng cấm")
plt.show()
