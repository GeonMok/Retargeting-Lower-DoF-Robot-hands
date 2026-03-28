# evaluate_trajectory.py
import numpy as np

FILE_DROP = "C:/4-1/KIAT/Preprocessing/allegro_trajectory_s1_apple_eat_1_drop.npy"
FILE_SIMPLE = "C:/4-1/KIAT/Preprocessing/allegro_trajectory_s1_apple_eat_1_simple.npy"

def compute_smoothness(trajectory: np.ndarray):
    """
    관절 각도의 3차 미분(Jerk)의 평균 절대값을 계산하여 부드러움을 평가합니다.
    값이 작을수록 움직임이 부드럽고 튀지 않음을 의미합니다.
    """
    # 손가락 관절(6~21번 인덱스) 데이터만 추출 (프레임, 16)
    finger_qpos = trajectory[:, 6:22]
    
    # 1차 미분 (속도)
    velocity = np.diff(finger_qpos, axis=0)
    # 2차 미분 (가속도)
    acceleration = np.diff(velocity, axis=0)
    # 3차 미분 (Jerk - 덜덜 떨림)
    jerk = np.diff(acceleration, axis=0)
    
    # 전체 프레임과 모든 관절에 대한 평균 절대값 계산
    mean_jerk = np.mean(np.abs(jerk))
    return mean_jerk

print("=== 정량 평가: Manipulation Smoothness (Jerk) ===")
try:
    traj_drop = np.load(FILE_DROP)
    traj_simple = np.load(FILE_SIMPLE)
    
    jerk_drop = compute_smoothness(traj_drop)
    jerk_simple = compute_smoothness(traj_simple)
    
    print(f"[M_drop]   Mean Jerk: {jerk_drop:.6f}")
    print(f"[M_simple] Mean Jerk: {jerk_simple:.6f}")
    
    if jerk_simple < jerk_drop:
        print("\n🏆 결론: M_simple (Synergy) 방식이 동작을 더 부드럽게(낮은 Jerk) 만듭니다!")
    else:
        print("\n결론: M_drop 방식의 Jerk가 더 낮습니다.")

except FileNotFoundError:
    print("🚨 궤적 파일을 찾을 수 없습니다. generate_baseline.py를 통해 두 파일을 먼저 생성해주세요.")