import torch
import torch.nn as nn
import torch.autograd as autograd

from py_diff_pd.core.py_diff_pd_core import StdRealVector, StdIntVector
from py_diff_pd.common.common import ndarray


class SimFunction(autograd.Function):
    @staticmethod
    def forward(ctx, env, dofs, act_dofs, method, q, v, x, f_ext, dt, option):

        ctx.env = env
        ctx.dofs = dofs
        ctx.act_dofs = act_dofs
        ctx.method = method
        ctx.dt = dt
        ctx.option = option
        
        frame_num = 1
        control_frame_num = frame_num
        chamber_num = len(env.chambers)
    
        def variable_to_act (x):
            # Input has shape [chamber_num // control_frequency] = [chamber_num * (frame_num // control_frame_num)]
            # Output actuation should be of shape [frame_num, act_elem_num]
            acts = []
            for i in range(len(x) // chamber_num):
                frame_act = []
                for k in range(chamber_num):
                    frame_act += [torch.ones(len(env.chambers[k])).to(x.device) * x[i*chamber_num + k]]

                frame_act = torch.cat(frame_act)
                acts += [torch.repeat_interleave(torch.unsqueeze(frame_act, dim=0), control_frame_num, dim=0)]

            ### We might have too many actuated frames, just trim the end as we took the ceil before for num_controllable.
            return torch.cat(acts, axis=0)[:frame_num]
		    
        a = variable_to_act(x)[0]
        
        # Store for backward
        ctx.mat_w_dofs = env.deformable().NumOfPdElementEnergies()
        ctx.act_w_dofs = env.deformable().NumOfPdMuscleEnergies()
        ctx.state_p_dofs = env.deformable().NumOfStateForceParameters()

        q_array = StdRealVector(q.double().cpu().detach().numpy())
        v_array = StdRealVector(v.double().cpu().detach().numpy())
        a_array = StdRealVector(a.double().cpu().detach().numpy())
        f_ext_array = StdRealVector(f_ext.double().cpu().detach().numpy())
        q_next_array = StdRealVector(dofs)
        v_next_array = StdRealVector(dofs)

        env.deformable().PyForward(
            method, q_array, v_array, a_array, f_ext_array, dt, option, q_next_array, v_next_array, StdIntVector(0))

        q_next = torch.as_tensor(ndarray(q_next_array))
        v_next = torch.as_tensor(ndarray(v_next_array))

        ctx.save_for_backward(x, q, v, a, f_ext, q_next, v_next)

        return q_next, v_next


    @staticmethod
    def backward(ctx, dl_dq_next, dl_dv_next):

        x, q, v, a, f_ext, q_next, v_next = ctx.saved_tensors
        dofs, act_dofs, mat_w_dofs, act_w_dofs, state_p_dofs = ctx.dofs, ctx.act_dofs, ctx.mat_w_dofs, ctx.act_w_dofs, ctx.state_p_dofs

        q_array = StdRealVector(q.double().cpu().detach().numpy())
        v_array = StdRealVector(v.double().cpu().detach().numpy())
        a_array = StdRealVector(a.double().cpu().detach().numpy())
        f_ext_array = StdRealVector(f_ext.double().cpu().detach().numpy())
        q_next_array = StdRealVector(q_next.cpu().detach().numpy())
        v_next_array = StdRealVector(v_next.cpu().detach().numpy())

        dl_dq_next_array = StdRealVector(dl_dq_next.cpu().detach().numpy())
        dl_dv_next_array = StdRealVector(dl_dv_next.cpu().detach().numpy())

        dl_dq_array = StdRealVector(dofs)
        dl_dv_array = StdRealVector(dofs)
        dl_da_array = StdRealVector(act_dofs)
        dl_df_ext_array = StdRealVector(dofs)
        #dl_dwi = StdRealVector(2)
        dl_dmat_wi = StdRealVector(mat_w_dofs)
        dl_dact_wi = StdRealVector(act_w_dofs)
        dl_dstate_pi = StdRealVector(state_p_dofs)
        
        
        ctx.env.deformable().PyBackward(
            ctx.method, q_array, v_array, a_array, f_ext_array, ctx.dt,
            q_next_array, v_next_array, StdIntVector(0), dl_dq_next_array, dl_dv_next_array, ctx.option,
            dl_dq_array, dl_dv_array, dl_da_array, dl_df_ext_array, dl_dmat_wi, dl_dact_wi, dl_dstate_pi)

        dl_dq = torch.as_tensor(ndarray(dl_dq_array))
        dl_dv = torch.as_tensor(ndarray(dl_dv_array))
        dl_da = torch.as_tensor(ndarray(dl_da_array))
        dl_df_ext = torch.as_tensor(ndarray(dl_df_ext_array))
        
        
        frame_num = 1
        control_frame_num = frame_num
        chamber_num = len(ctx.env.chambers)
        
        def variable_to_gradient(x, dl_dact):
            import numpy as np  
            dl_dact = ndarray(dl_dact)
            grad = np.zeros_like(x)
            for i in range(len(x) // chamber_num):
                done_chamber_lens = 0    # When we dissect the actuation gradients into all the separate chambers
                for k in range(chamber_num):
                    for f in range(control_frame_num):
                        f_idx = i * control_frame_num + f
                        if f_idx >= frame_num:
		                    # When we cut off the number of actuated frames
                            break
                        #grad_act = dl_dact[f_idx]  # We're going frame by frame, so we don't want indexing
                        grad_act = dl_dact

                        grad[i*chamber_num+k] += np.sum(grad_act[done_chamber_lens:done_chamber_lens+len(ctx.env.chambers[k])])
                    done_chamber_lens += len(ctx.env.chambers[k])

            return grad
		    
        dl_dx = variable_to_gradient(x, dl_da)
        

        return (None, None, None, None,
            torch.as_tensor(ndarray(dl_dq)),
            torch.as_tensor(ndarray(dl_dv)),
            torch.as_tensor(ndarray(dl_dx)),
            torch.as_tensor(ndarray(dl_df_ext)),
            None, None)


class Sim(nn.Module):
    def __init__(self, env):
        super(Sim, self).__init__()
        self.env = env

    def forward(self, dofs, act_dofs, method, q, v, a, f_ext, dt, option):
        return SimFunction.apply(
            self.env, dofs, act_dofs, method, q, v, a, f_ext, dt, option)
