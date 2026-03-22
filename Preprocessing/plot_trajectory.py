import numpy as np
import matplotlib.pyplot as plt

# 1. 저장된 궤적 데이터 불러오기
trajectory = np.load("allegro_trajectory_s1_apple_eat_1.npy")
frames = trajectory.shape[0]

# 2. 로봇 손가락 16개 각도만 추출 (앞의 6개는 손목 이동/회전이므로 제외)
joint_angles = trajectory[:, 6:22]

# 3. 그래프 그리기 (4개의 손가락 중 대표로 검지손가락 4개 모터만 확인)
# Allegro Hand 검지(Index) 인덱스: 4, 5, 6, 7번 모터
plt.figure(figsize=(12, 6))
plt.plot(range(frames), joint_angles[:, 4], label="Index Joint 1 (Base)")
plt.plot(range(frames), joint_angles[:, 5], label="Index Joint 2")
plt.plot(range(frames), joint_angles[:, 6], label="Index Joint 3")
plt.plot(range(frames), joint_angles[:, 7], label="Index Joint 4 (Tip)")

plt.title("Allegro Hand Index Finger Joint Trajectory (1378 Frames)")
plt.xlabel("Frame")
plt.ylabel("Joint Angle (Radian)")
plt.legend()
plt.grid(True)
plt.show()