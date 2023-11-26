import numpy as np
import gym
from gym import spaces
import pygame




class WareHouseEnv(gym.Env):
    def __init__(self, map_size=5, max_steps=2000, graphic=True, fps = 150):
        super(WareHouseEnv, self).__init__()
        ## 학습
        self.episode = 0
        self.map_size = map_size
        self.log_map = []

        self.max_steps = max_steps
        self.current_step = 0                                                                                                      
        
        # Actions for each vehicle: 0:up, 1:right, 2:down, 3:left
        self.action_space = spaces.MultiDiscrete([5] * (2 + 3))
        self.observation_space = spaces.Box(low=0, high=5, shape=(map_size, map_size), dtype=np.int32)

        self.desired_total_sum = 150
        self.cargo_map = self.generate_fixed_sum_array((map_size, map_size), self.desired_total_sum)
        # self.cargo_map = np.random.randint(0, 11, (map_size, map_size))
        self.cargo_map[0][0] = 0  # soil_bank position

        self.num_lift = 2
        self.num_robot = 3

        # Initialize the positions, can be randomized
        self.lift_positions = [self.random_position() for _ in range(self.num_lift)]
        self.robot_positions = [self.random_position() for _ in range(self.num_robot)]
        
        self.lift_carry = [0] * self.num_lift
        self.robot_load = [0] * self.num_robot
        self.reward = 0

        ## render_frame
        self.window = None
        self.clock = None
        self.draw_fps = fps
        self.cell_size = 90
        self.win_size = self.cell_size * self.map_size
        self.graphic = graphic

    def reset(self):
        self.current_step = 0
        self.cargo_map = self.generate_fixed_sum_array((self.map_size, self.map_size), self.desired_total_sum)
        self.cargo_map[0][0] = 0
        self.lift_positions = [self.random_position() for _ in range(self.num_lift)]
        self.robot_positions = [self.random_position() for _ in range(self.num_robot)]
        self.lift_carry = [0] * self.num_lift
        self.robot_load = [0] * self.num_robot
        self.reward = 0
        return self._next_observation()

    def _next_observation(self):
        return self.cargo_map

    def step(self, actions):

        # 초기화 보상
        reward = 0
        lift_penalty = 0
        robot_penalty = 0
        lift_bonus = 0
        
        # Move each excavator based on its action
        for i in range(self.num_lift):
            action = actions[i]
            if self.lift_carry[i] != 1:
                prev_position = list(self.lift_positions[i])
                if action == 0:  # up
                    self.lift_positions[i][1] = min(self.map_size-1, self.lift_positions[i][1] + 1)
                elif action == 1:  # right
                    self.lift_positions[i][0] = min(self.map_size-1, self.lift_positions[i][0] + 1)
                elif action == 2:  # down
                    self.lift_positions[i][1] = max(0, self.lift_positions[i][1] - 1)
                elif action == 3:  # left
                    self.lift_positions[i][0] = max(0, self.lift_positions[i][0] - 1)
                else:
                    pass
                # excavator가 carry중인데, 덤프트럭이 존재하면 penalty값 감소
                if self.lift_positions[i] in self.robot_positions and self.lift_carry[i] == 1:
                    lift_penalty -= 2

        # Move each dumptruck based on its action
        for i in range(self.num_robot):
            action = actions[self.num_lift + i]
            prev_position = list(self.robot_positions[i])
            if action == 0:  # up
                self.robot_positions[i][1] = min(self.map_size-1, self.robot_positions[i][1] + 1)
            elif action == 1:  # right
                self.robot_positions[i][0] = min(self.map_size-1, self.robot_positions[i][0] + 1)
            elif action == 2:  # down
                self.robot_positions[i][1] = max(0, self.robot_positions[i][1] - 1)
            elif action == 3:  # left
                self.robot_positions[i][0] = max(0, self.robot_positions[i][0] - 1)
            else:
                pass

            # 불필요한 움직임 패널티 주기(제자리에 머무는 경우와 사토장으로 이동하지 않았을 때, 패널티 줌)
            # if self.robot_positions[i] == prev_position:
            #     robot_penalty -= 1
            if self.robot_load[i] >= 5 and self.robot_positions[i] != [0, 0]:
                robot_penalty += 3

        # 로직: 굴삭 및 로딩
        robot_positions_set = set(tuple(pos) for pos in self.robot_positions)  # 덤프트럭 위치를 set으로 변환

        for i in range(self.num_lift):
            if self.lift_carry[i] == 1:
                if tuple(self.lift_positions[i]) in robot_positions_set:  # O(1)의 시간 복잡도로 위치 확인
                    j = self.robot_positions.index(self.lift_positions[i])  # 일치하는 덤프트럭의 인덱스를 찾습니다.
                    if self.robot_load[j] < 6:
                        self.robot_load[j] += 1
                        self.lift_carry[i] = 0
                        lift_bonus += 5  # 굴삭 및 로딩 보상 증가
            if self.lift_carry[i] == 0 and self.cargo_map[self.lift_positions[i][1]][self.lift_positions[i][0]] > 0:
                self.lift_carry[i] = 1
                self.cargo_map[self.lift_positions[i][1]][self.lift_positions[i][0]] -= 1

        for j in range(self.num_robot):
            if self.robot_positions[j] == [0, 0]:
                reward += self.robot_load[j] * 1.2  # 추가적인 로딩 보상
                self.robot_load[j] = 0


        # 패널티 및 보상 적용
        self.reward += reward - lift_penalty - robot_penalty + lift_bonus
        if np.sum(self.cargo_map) == 0:
            self.reward += 5000 - self.current_step  # 스텝 수에 따른 보상 감소
            done = True
            
            ## 출력 2가지 스타일
            if self.episode % 100 == 0:
                print(f"Episode {self.episode}. Steps taken: {self.current_step}. Remaining soil: {np.sum(self.cargo_map)}. Total reward: {self.reward}")
            # print(f"Episode {self.episode}. Steps taken: {self.current_step}. Remaining soil: {np.sum(self.cargo_map)}. Total reward: {self.reward}")
            self.episode += 1
        else:
            done = False

        
        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True
            self.episode += 1

        return self._next_observation(), self.reward, done, {}

    def generate_fixed_sum_array(self, shape, total_sum):
        # 랜덤한 배열 생성
        arr = np.random.rand(*shape)
        
        # 배열의 총합으로 나누어 비율을 계산
        ratio = arr / arr.sum()
        
        # 원하는 총합에 맞게 배열을 조정
        arr = (ratio * total_sum).astype(int)
        
        # 반올림 오차로 인해 조금 더하거나 빼게 되는 경우, 그 차이를 보정
        diff = total_sum - arr.sum()
        if diff > 0:
            indices = np.random.choice(np.arange(shape[0] * shape[1]), diff, replace=False)
            for idx in indices:
                i, j = np.unravel_index(idx, shape)
                arr[i, j] += 1
        elif diff < 0:
            indices = np.random.choice(np.arange(shape[0] * shape[1]), -diff, replace=False)
            for idx in indices:
                i, j = np.unravel_index(idx, shape)
                arr[i, j] -= 1

        return arr