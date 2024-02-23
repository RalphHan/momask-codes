from smplx import SMPL
from smplx.utils import SMPLOutput, Tensor
from smplx.lbs import vertices2joints, blend_shapes, batch_rodrigues, transform_mat
from typing import Optional, Tuple
import torch


def batch_rigid_transform(
        rot_mats: Tensor,
        joints: Tensor,
        parents: Tensor,
) -> Tensor:
    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)
    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]],
                                transforms_mat[:, i])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=1)
    posed_joints = transforms[:, :, :3, 3]
    return posed_joints


def lbs(
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        J_regressor: Tensor,
        parents: Tensor,
        pose2rot: bool = True,
) -> Tuple[Tensor, Tensor]:
    batch_size = max(betas.shape[0], pose.shape[0])
    v_shaped = v_template + blend_shapes(betas, shapedirs)
    J = vertices2joints(J_regressor, v_shaped)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view(
            [batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view(batch_size, -1, 3, 3)
    J_transformed = batch_rigid_transform(rot_mats, J, parents)
    return J_transformed


class MySMPL(SMPL):
    def forward(
            self,
            betas: Optional[Tensor] = None,
            body_pose: Optional[Tensor] = None,
            global_orient: Optional[Tensor] = None,
            transl: Optional[Tensor] = None,
            return_verts=True,
            return_full_pose: bool = False,
            pose2rot: bool = True,
            **kwargs
    ) -> SMPLOutput:
        ''' Forward pass for the SMPL model

            Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)
            betas: torch.tensor, optional, shape BxN_b
                If given, ignore the member variable `betas` and use it
                instead. For example, it can used if shape parameters
                `betas` are predicted from some external model.
                (default=None)
            body_pose: torch.tensor, optional, shape Bx(J*3)
                If given, ignore the member variable `body_pose` and use it
                instead. For example, it can used if someone predicts the
                pose of the body joints are predicted from some external model.
                It should be a tensor that contains joint rotations in
                axis-angle format. (default=None)
            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            return_verts: bool, optional
                Return the vertices. (default=True)
            return_full_pose: bool, optional
                Returns the full axis-angle pose vector (default=False)

            Returns
            -------
        '''
        global_orient = (global_orient if global_orient is not None else
                         self.global_orient)
        body_pose = body_pose if body_pose is not None else self.body_pose
        betas = betas if betas is not None else self.betas
        apply_trans = transl is not None or hasattr(self, 'transl')
        if transl is None and hasattr(self, 'transl'):
            transl = self.transl
        full_pose = torch.cat([global_orient, body_pose], dim=1)
        batch_size = max(betas.shape[0], global_orient.shape[0],
                         body_pose.shape[0])
        if betas.shape[0] != batch_size:
            num_repeats = int(batch_size / betas.shape[0])
            betas = betas.expand(num_repeats, -1)
        joints = lbs(betas, full_pose, self.v_template,
                     self.shapedirs,
                     self.J_regressor, self.parents, pose2rot=pose2rot)
        if self.joint_mapper is not None:
            joints = self.joint_mapper(joints)
        if apply_trans:
            joints += transl.unsqueeze(dim=1)
        output = SMPLOutput(vertices=None,
                            global_orient=global_orient,
                            body_pose=body_pose,
                            joints=joints,
                            betas=betas,
                            full_pose=full_pose if return_full_pose else None)
        return output
