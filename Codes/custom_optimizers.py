# custom_optimizers.py
import numpy as np
import torch
import torch.nn.functional as F
from dex_retargeting.optimizer import PositionOptimizer, VectorOptimizer

class MyCustomSDFOptimizer(PositionOptimizer):
    def __init__(self, sdf_path: str, penetration_threshold=0.005, penetration_weight=2000.0, *args, **kwargs):
        """
        True SDF 기반의 충돌 방지 Loss가 반영된 커스텀 옵티마이저입니다.
        
        Args:
            sdf_path: bake_sdf.py로 구워진 .pt 파일 경로
            penetration_threshold: 파고듦 허용 임계값 (m), 기본 5mm
            penetration_weight: 충돌 loss의 강력한 가중치
        """
        super().__init__(*args, **kwargs)
        
        # 1. 구워진 SDF 데이터 로드
        print(f"\n🛡️ Loading Baked SDF from: {sdf_path}...")
        sdf_data = torch.load(sdf_path)
        
        # PyTorch grid_sample을 위해 GPU가 있다면 GPU로 올리는 것이 좋습니다. (여기선 CPU 가정)
        self.sdf_grid = sdf_data['sdf_grid'] # shape: (1, Res, Res, Res)
        self.sdf_res = sdf_data['resolution']
        self.grid_dim = sdf_data['grid_dim_meters']
        
        # 충돌 하이퍼파라미터 저장
        self.penetration_threshold = penetration_threshold
        self.penetration_weight = penetration_weight
        
        # 매 프레임 물체의 포즈를 받아올 변수 (텐서 형태)
        self.current_obj_transl = None
        self.current_obj_rot_matrix = None
        
        print(f"-> SDF Optimizer ready. (Object: {sdf_data['obj_name']})")

    def update_object_pose(self, transl: np.ndarray, rot_matrix: np.ndarray):
        """매 프레임 최적화 시작 전 물체의 포즈 업데이트"""
        self.current_obj_transl = torch.tensor(transl, dtype=torch.float32)
        self.current_obj_rot_matrix = torch.tensor(rot_matrix, dtype=torch.float32)

    def get_objective_function(self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # 1. Kinematics: 로봇 손끝 월드 좌표(torch_body_pos) 계산
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [self.robot.get_link_pose(index) for index in self.target_link_indices]
            body_pos = np.stack([pose[:3, 3] for pose in target_link_poses], axis=0)

            # PyTorch 텐서 변환 및 미분 활성화
            torch_body_pos = torch.as_tensor(body_pos, dtype=torch.float32)
            torch_body_pos.requires_grad_()
            
            # ==========================================================
            # 💡 [핵심 수술: True SDF / Penetration Loss]
            # ==========================================================
            sdf_loss = torch.tensor(0.0, requires_grad=True)

            if self.current_obj_transl is not None:
                num_fingers = torch_body_pos.shape[0]
                
                # [수학 핵심 1: 월드 좌표 -> 물체 좌표계 변환]
                # SDF 그리드는 물체 좌표계 기준이므로 손끝 좌표를 역변환해야 합니다.
                # (손끝W - 물체T) @ 물체R
                fingertips_obj_space = (torch_body_pos - self.current_obj_transl) @ self.current_obj_rot_matrix

                # [수학 핵심 2: 물체 좌표 -> SDF 그리드 인덱스(-1 ~ 1) 변환]
                # torch.grid_sample은 좌표가 -1 ~ 1 사이여야 합니다. 
                # 우리 그리드는 정육면체이고 한 변이 grid_dim이므로, grid_dim/2 로 나눠줍니다.
                grid_half_dim = self.grid_dim / 2.0
                fingertips_grid_idx = fingertips_obj_space / grid_half_dim
                
                # grid_sample 요구 형식에 맞게 리쉐이프 (Batch=1, N_Points, 1, 1, 3_Coords)
                # mesh_to_sdf 그리드(ij)와 grid_sample(xyz)의 축 순서를 맞춰줍니다 (x, y, z -> xyz)
                grid_input = fingertips_grid_idx.view(1, num_fingers, 1, 1, 3)
                
                # [수학 핵심 3: Trilinear Interpolation (삼선형 보간)]
                # 버터처럼 부드럽고 연속적인 SDF 값과 기울기를 뽑아냅니다!
                # 🚨 주의: mesh_to_sdf는 ij indexing을 쓰므로 grid_sample에서 align_corners=True 필수!
                sampled_sdf = F.grid_sample(
                    self.sdf_grid.to(torch_body_pos.device), # SDF 그리드 텐서
                    grid_input,
                    mode='bilinear', # 3D에서는 trilinear로 동작함
                    padding_mode='border', # 그리드 밖은 경계값 사용 (여유 패딩이 있어서 안전함)
                    align_corners=True 
                )
                
                # 결과 샘플링 값 추출 (4,) - 내부(-), 외부(+)
                sdf_values = sampled_sdf.view(num_fingers)

                # [수학 핵심 4: 충돌 조건 및 Loss 계산]
                # SDF 값이 임계값(penetration_threshold, 기본 5mm)보다 작으면 파고든 것으로 간주.
                # 예: 물체 내부(-1cm)에 있다면 sdf_values=-0.01, violation = 0.005 - (-0.01) = 0.015 (1.5cm 파고듦)
                violation = self.penetration_threshold - sdf_values
                
                # ReLU를 이용해 파고든 경우(violation > 0)에만 Loss 부여
                active_penetration = F.relu(violation)
                
                # 파고든 깊이의 제곱을 가중치와 곱해서 합산 (기하급수적 방어)
                sdf_loss = torch.sum(active_penetration ** 2) * self.penetration_weight

            # ==========================================================
            
            # (1) 기존 Loss: Kinematic Error
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            
            # 최종 통합 Loss!
            total_loss = huber_distance + sdf_loss

            result = total_loss.cpu().detach().item()

            if grad.size > 0:
                # ... (이후 자코비안 계산, total_loss.backward(), 체인 룰 등은 이전과 완벽히 동일) ...
                
                # 단 한줄, total_loss.backward() 가 실행될 때,
                # torch.grid_sample 내부의 수학 공식에 의해 버터처럼 부드러운
                # SDF 충돌 방지 기울기(Gradient)가 qpos로 전파됩니다!
                
                # ... (이하 생략) ...
                # (이전 PositionOptimizer 코드의 grad[:] 계산 부분 그대로 사용)

class MyCustomVectorOptimizer(VectorOptimizer):
    def get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        # TODO: SDF Loss, Limit Loss, etc.
        
        return super().get_objective_function(target_vector, fixed_qpos, last_qpos)