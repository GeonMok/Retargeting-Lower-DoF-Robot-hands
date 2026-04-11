# custom_optimizers.py
import numpy as np
import torch
import torch.nn.functional as F
from dex_retargeting.optimizer import PositionOptimizer, VectorOptimizer

class MyCustomPositionOptimizer(PositionOptimizer):
    def __init__(self, sdf_path: str = None, 
                 use_sdf=True, penetration_threshold=0.005, penetration_weight=2000.0,
                 use_attraction=True, contact_margin=0.01, attraction_weight=500.0,
                 use_limit=True, limit_margin=0.05, limit_weight=50.0,
                 *args, **kwargs):
        """        
        Args:
            sdf_path: .pt file path baked by bake_sdf.py
            penetration_threshold: default 5mm (0.005m)
            penetration_weight: collision weight for the penetration loss
        """
        super().__init__(*args, **kwargs)
        self.use_limit = use_limit
        self.limit_margin = limit_margin
        self.limit_weight = limit_weight
        
        if self.use_limit:
            limits = self.robot.joint_limits[self.idx_pin2target]
            self.q_min = limits[:, 0]
            self.q_max = limits[:, 1]
            print(f"🟢 [Limit Term]: ON (Margin: {self.limit_margin} rad, Weight: {self.limit_weight})")
        else:
            print("🔴 [Limit Term]: OFF")

        self.use_sdf = use_sdf
        self.use_attraction = use_attraction
        self.contact_margin = contact_margin
        self.attraction_weight = attraction_weight

        self.current_obj_transl = None
        self.current_obj_rot_matrix = None
        
        if self.use_sdf and sdf_path is not None:
            print(f"🟢 [SDF Term]: ON (Loading from {sdf_path}...)")
            sdf_data = torch.load(sdf_path, weights_only=False)
            
            self.sdf_grid = sdf_data['sdf_grid'].unsqueeze(0) 
            self.penetration_threshold = penetration_threshold
            self.penetration_weight = penetration_weight
            
            self.old_center = torch.tensor(sdf_data['old_center'], dtype=torch.float32)
            self.old_scale = float(sdf_data['old_scale'])
            self.limit = 0.5 + float(sdf_data['padding_rate']) 
            self.sdf_res = sdf_data['resolution']
            self.grid_dim = sdf_data['grid_dim_meters']
            self.is_grid_on_device = False
            print(f"   -> SDF Ready. (Object: {sdf_data['obj_name']})")
        else:
            self.use_sdf = False
            print("🔴 [SDF Term]: OFF")

    def update_object_pose(self, transl: np.ndarray, rot_matrix: np.ndarray):
        """Every frame, we will update the current object pose (translation + rotation) to compute the SDF loss correctly."""
        if self.use_sdf:
            self.current_obj_transl = torch.tensor(transl, dtype=torch.float32)
            self.current_obj_rot_matrix = torch.tensor(rot_matrix, dtype=torch.float32)

    def get_objective_function(self, target_pos: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        qpos = np.zeros(self.num_joints)
        qpos[self.idx_pin2fixed] = fixed_qpos
        torch_target_pos = torch.as_tensor(target_pos)
        torch_target_pos.requires_grad_(False)

        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos[self.idx_pin2target] = x

            # [Forward Kinematics]
            if self.adaptor is not None:
                qpos[:] = self.adaptor.forward_qpos(qpos)[:]

            self.robot.compute_forward_kinematics(qpos)
            target_link_poses = [self.robot.get_link_pose(index) for index in self.target_link_indices]
            body_pos = np.stack([pose[:3, 3] for pose in target_link_poses], axis=0)

            # PyTorch Computation 
            torch_body_pos = torch.as_tensor(body_pos, dtype=torch.float32)
            torch_body_pos.requires_grad_()
            
            # ==========================================================
            # 💡 Custom Loss1: [True SDF / Penetration Loss]
            # ==========================================================
            sdf_loss = torch.tensor(0.0, requires_grad=True)
            attraction_loss = torch.tensor(0.0, requires_grad=True)

            if self.use_sdf and self.current_obj_transl is not None:
                if not self.is_grid_on_device:
                    self.sdf_grid = self.sdf_grid.to(torch_body_pos.device)
                    self.old_center = self.old_center.to(torch_body_pos.device)
                    self.is_grid_on_device = True

                num_fingers = torch_body_pos.shape[0]
                device = torch_body_pos.device
                
                fingertips_raw_local = (torch_body_pos - self.current_obj_transl) @ self.current_obj_rot_matrix
                fingertips_normalized = (fingertips_raw_local - self.old_center.to(device)) / self.old_scale
                fingertips_grid_idx = fingertips_normalized / self.limit
                
                grid_input = fingertips_grid_idx.view(1, num_fingers, 1, 1, 3)
                grid_input = torch.flip(grid_input, dims=[-1]) 
                
                sampled_sdf = F.grid_sample(
                    self.sdf_grid, grid_input, mode='bilinear', padding_mode='border', align_corners=True 
                )
                
                sdf_values = sampled_sdf.view(num_fingers)
                violation = self.penetration_threshold - sdf_values
                active_penetration = F.relu(violation)
                sdf_loss = torch.sum(active_penetration ** 2) * self.penetration_weight
                if self.use_attraction:
                    floating = sdf_values - self.contact_margin
                    active_floating = F.relu(floating)
                    attraction_loss = torch.sum(active_floating ** 2) * self.attraction_weight

            # ==========================================================
            # 💡 [Loss 2]: Limit Loss (NumPy)
            # ==========================================================
            limit_loss_val = 0.0
            grad_limit_val = np.zeros_like(x)

            if self.use_limit:
                violation_lower = np.zeros_like(x)
                violation_upper = np.zeros_like(x)
                
                valid_lower = np.isfinite(self.q_min)
                violation_lower[valid_lower] = np.maximum(0.0, self.q_min[valid_lower] + self.limit_margin - x[valid_lower])
                
                valid_upper = np.isfinite(self.q_max)
                violation_upper[valid_upper] = np.maximum(0.0, x[valid_upper] - (self.q_max[valid_upper] - self.limit_margin))
                
                limit_loss_val = np.sum(violation_lower**2 + violation_upper**2) * self.limit_weight
                grad_limit_val = 2.0 * self.limit_weight * (violation_upper - violation_lower)
            # ==========================================================
            
            huber_distance = self.huber_loss(torch_body_pos, torch_target_pos)
            
            total_loss = huber_distance + sdf_loss + attraction_loss

            result = total_loss.cpu().detach().item() + limit_loss_val

            if grad.size > 0:
                jacobians = []
                for i, index in enumerate(self.target_link_indices):
                    link_body_jacobian = self.robot.compute_single_link_local_jacobian(qpos, index)[:3, ...]
                    link_rot = target_link_poses[i][:3, :3]
                    link_kinematics_jacobian = link_rot @ link_body_jacobian
                    jacobians.append(link_kinematics_jacobian)

                jacobians = np.stack(jacobians, axis=0)
                
                total_loss.backward()
                grad_pos = torch_body_pos.grad.cpu().numpy()[:, None, :]

                if self.adaptor is not None:
                    jacobians = self.adaptor.backward_jacobian(jacobians)
                else:
                    jacobians = jacobians[..., self.idx_pin2target]

                grad_qpos = np.matmul(grad_pos, jacobians)
                grad_qpos = grad_qpos.mean(1).sum(0)
                grad_qpos += 2 * self.norm_delta * (x - last_qpos)

                grad_qpos += grad_limit_val
                grad[:] = grad_qpos[:]

            return result
        
        return objective

class MyCustomVectorOptimizer(VectorOptimizer):
    def get_objective_function(self, target_vector: np.ndarray, fixed_qpos: np.ndarray, last_qpos: np.ndarray):
        return super().get_objective_function(target_vector, fixed_qpos, last_qpos)