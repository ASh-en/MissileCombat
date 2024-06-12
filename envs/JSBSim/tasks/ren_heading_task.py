import numpy as np
from typing import List, Tuple
from gymnasium import spaces
from .task_base import BaseTask
from ..core.catalog import Catalog as c
from ..reward_functions import AltitudeReward, HeadingReward, TimeoutReward
from ..termination_conditions import ExtremeState, LowAltitude, Overload, Timeout, UnreachHeading
from ..utils.utils import body2ned, ned2body

class RenHeadingTask(BaseTask):
    '''
    Control target heading with discrete action space
    '''
    def __init__(self, config):

        self.type = "body"

        super().__init__(config)
        # self.use_noise = getattr(self.config, 'use_noise', False)
        self.use_noise = False
        self.use_ekf = getattr(self.config, 'use_ekf', False)
        self.use_data_loss = getattr(self.config, 'use_data_loss', False)
        self.use_data_loss = True
        self.data_loss_prop = getattr(self.config, 'data_loss_prop', 0.02)  # type: float
        self.std = [50., 0., 10., 10., 10.]    # altitude  (unit: m),  attitude_psi_deg (unit: deg) ,
                                               # v_body_x   (unit: m/s), v_body_y   (unit: m/s), v_body_z   (unit: m/s)

        self.reward_functions = [
            HeadingReward(self.config),
            AltitudeReward(self.config),
            TimeoutReward(self.config),
        ]
        self.termination_conditions = [
            UnreachHeading(self.config),
            ExtremeState(self.config),
            Overload(self.config),
            LowAltitude(self.config),
            Timeout(self.config),
        ]

    @property
    def num_agents(self):
        return 1

    def load_variables(self):
        self.state_var = [
            c.delta_altitude,
            c.delta_heading,
            c.delta_velocities_u,
            c.position_h_sl_m,
            c.attitude_pitch_rad,
            c.attitude_roll_rad,
            c.velocities_u_mps,
            c.velocities_v_mps,
            c.velocities_w_mps,
            c.velocities_p_rad_sec,
            c.velocities_q_rad_sec,
            c.velocities_r_rad_sec,
            c.fcs_left_aileron_pos_norm,
            c.fcs_right_aileron_pos_norm,
            c.fcs_elevator_pos_norm,
            c.fcs_rudder_pos_norm,
            c.aero_beta_deg
        ]
        self.action_var = [
            c.fcs_aileron_cmd_norm,             # [-1., 1.]
            c.fcs_elevator_cmd_norm,            # [-1., 1.]
            c.fcs_rudder_cmd_norm,              # [-1., 1.]
            c.fcs_throttle_cmd_norm,            # [0.4, 0.9]
        ]
        self.render_var = [
            c.position_long_gc_deg,
            c.position_lat_geod_deg,
            c.position_h_sl_m,
            c.attitude_roll_rad,
            c.attitude_pitch_rad,
            c.attitude_heading_true_rad,
        ]

    def load_observation_space(self):
        self.observation_space = spaces.Box(low=-10, high=10., shape=(17,))

    def load_action_space(self):
        # aileron, elevator, rudder, throttle
        self.action_space = spaces.MultiDiscrete([41, 41, 41, 30])


    def get_termination(self, env, agent_id, info={}) -> Tuple[bool, dict]:
        """
        Aggregate termination conditions

        Args:
            env: environment instance
            agent_id: current agent id
            info: additional info

        Returns:
            (tuple):
                done(bool): whether the episode has terminated
                info(dict): additional info
        """
        done = False
        success = True
        info['termination'] = 0
        for condition in self.termination_conditions:
            d, s, info = condition.get_termination(self, env, agent_id, info)
            done = done or d
            success = success and s

            if done:
                info['heading_turn_counts'] = env.heading_turn_counts
                info['end_step'] = env.current_step
                break
        if info['termination'] == 0:
            del info['termination']
        return done, info

    def get_obs(self, env, agent_id):
        """
        Convert simulation states into the format of observation_space.


        """
        obs = np.array(env.agents[agent_id].get_property_values(self.state_var))
        norm_obs = np.zeros(17)
        if self.use_noise:
            # delta_altitude = target_altitude - position_h_sl
            noise = np.random.normal(0, self.std)

        if self.use_data_loss:
            not_loss = np.array(np.random.binomial(1, 1-self.data_loss_prop, size=11))
            # obs = obs * not_loss

        norm_obs[0] = obs[0] / 1000             # 0. ego delta altitude (unit: 1km)
        norm_obs[1] = obs[1] / 180 * np.pi      # 1. ego delta heading  (unit rad)
        norm_obs[2] = obs[2] / 340              # 2. ego delta velocities_u (unit: mh)
        norm_obs[3] = obs[3] / 5000             # 3. ego_altitude   (unit: 5km)
        norm_obs[4] = obs[4]                    # 4. ego_pitch      (unit rad)
        norm_obs[5] = obs[5]                    # 5. ego_roll       (unit rad)
        norm_obs[6] = obs[6] / 340              # 6. body_v_x       (unit: mh)
        norm_obs[7] = obs[7] / 340              # 7. body_v_y       (unit: mh)
        norm_obs[8] = obs[8] / 340              # 8. body_v_z       (unit: mh)
        norm_obs[9] = obs[9]                    # 9. p              (unit: rad/s)
        norm_obs[10] = obs[10]                  # 10. q             (unit: rad/s)
        norm_obs[11] = obs[11]                  # 11. r             (unit: rad/s)
        norm_obs[12] = obs[12]                  # 12. left_aileron_pos_norm
        norm_obs[13] = obs[13]                  # 13. right_aileron_pos_norm
        norm_obs[14] = obs[14]                  # 14. elevator_pos_norm
        norm_obs[15] = obs[15]                  # 15. rudder_pos_norm
        norm_obs[16] = obs[16] / 180 * np.pi    # 16. aero_beta     (unit rad)

        norm_obs = np.clip(norm_obs, self.observation_space.low, self.observation_space.high)
        return norm_obs

    def normalize_action(self, env, agent_id, action):
        """Convert discrete action index into continuous value.
        """
        norm_act = np.zeros(4)
        norm_act[0] = action[0] * 2. / (self.action_space.nvec[0] - 1.) - 1.
        norm_act[1] = action[1] * 2. / (self.action_space.nvec[1] - 1.) - 1.
        norm_act[2] = action[2] * 2. / (self.action_space.nvec[2] - 1.) - 1.
        norm_act[3] = action[3] * 0.5 / (self.action_space.nvec[3] - 1.) + 0.4
        return norm_act

