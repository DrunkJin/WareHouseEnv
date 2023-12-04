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
        
        self.num_lift = 2
        self.num_robot = 3

        # Actions for each vehicle: 0:up, 1:right, 2:down, 3:left
        total_action_combinations = 4 ** (self.num_lift + self.num_robot) 
        self.action_space = spaces.Discrete(total_action_combinations)
        self.observation_space = spaces.Box(low=0, high=5, shape=(map_size, map_size), dtype=np.int32)

        self.desired_total_sum = 150
        self.cargo_map = self.generate_fixed_sum_array((map_size, map_size), self.desired_total_sum)
        # self.cargo_map = np.random.randint(0, 11, (map_size, map_size))
        self.cargo_map[0][0] = 0  # soil_bank position

        # Initialize the positions, can be randomized
        self.lift_positions = [[2, 2], [2, 3]]
        self.robot_positions = [[3, 3], [3, 2], [3, 1]]
        
        self.lift_carry = [0] * self.num_lift
        self.robot_load = [0] * self.num_robot
        self.reward = 0

        ## render_frame
        self.render_mode = "human"
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
        self.lift_carry = [0] * self.num_lift
        self.robot_load = [0] * self.num_robot
        self.reward = 0
        return self._next_observation()

    def _next_observation(self):
        return self.cargo_map

    def step(self, actions):
        individual_actions = []
        for _ in range(self.num_lift + self.num_robot):
            individual_actions.append(actions % 4)
            actions //= 4
        individual_actions.reverse()

        # 초기화 보상
        reward = 0
        lift_penalty = 0
        robot_penalty = 0
        lift_bonus = 0
        
        # Apply actions to lifts
        for i in range(self.num_lift):
            action = individual_actions[i]
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
                
                # excavator가 carry중인데, 덤프트럭이 존재하면 penalty값 감소
                # if self.lift_positions[i] in self.robot_positions and self.lift_carry[i] == 1:
                #     lift_penalty -= 2

        # Move each dumptruck based on its action
        for i in range(self.num_robot):
            action = individual_actions[self.num_lift + i]
            prev_position = list(self.robot_positions[i])
            if action == 0:  # up
                self.robot_positions[i][1] = min(self.map_size-1, self.robot_positions[i][1] + 1)
            elif action == 1:  # right
                self.robot_positions[i][0] = min(self.map_size-1, self.robot_positions[i][0] + 1)
            elif action == 2:  # down
                self.robot_positions[i][1] = max(0, self.robot_positions[i][1] - 1)
            elif action == 3:  # left
                self.robot_positions[i][0] = max(0, self.robot_positions[i][0] - 1)

            # 불필요한 움직임 패널티 주기(제자리에 머무는 경우와 사토장으로 이동하지 않았을 때, 패널티 줌)
            # if self.robot_load[i] >= 5 and self.robot_positions[i] != [0, 0]:
            #     robot_penalty += 3
            # elif self.cargo_map[self.robot_positions[i][1]][self.robot_positions[i][0]] == 0:
            #     robot_penalty -= 1  # 로봇이 이동한 위치에 짐이 없는 경우 패널티 감소
           

        # 로직: 굴삭 및 로딩
        robot_positions_set = set(tuple(pos) for pos in self.robot_positions)  # 덤프트럭 위치를 set으로 변환

        for i in range(self.num_lift):
            if self.lift_carry[i] == 1:
                if tuple(self.lift_positions[i]) in robot_positions_set:  # O(1)의 시간 복잡도로 위치 확인
                    j = self.robot_positions.index(self.lift_positions[i])  # 일치하는 덤프트럭의 인덱스를 찾습니다.
                    if self.robot_load[j] < 5:
                        self.robot_load[j] += 1
                        self.lift_carry[i] = 0
                        lift_bonus += 5  # 굴삭 및 로딩 보상 증가
            if self.lift_carry[i] == 0 and self.cargo_map[self.lift_positions[i][1]][self.lift_positions[i][0]] > 0:
                # 수정된 부분 시작
                self.lift_carry[i] = 1
                self.cargo_map[self.lift_positions[i][1]][self.lift_positions[i][0]] -= 1


                # if self.lift_positions[i] in self.robot_positions and self.lift_carry[i] == 1:
                #     lift_penalty += 2
                # else:
                #     self.lift_carry[i] = 1
                #     self.cargo_map[self.lift_positions[i][1]][self.lift_positions[i][0]] -= 1
                # 수정된 부분 끝

        for j in range(self.num_robot):
            if self.robot_positions[j] == [0, 0]:
                reward += self.robot_load[j] * 1.2  # 추가적인 로딩 보상
                self.robot_load[j] = 0


        # 패널티 및 보상 적용
        # self.reward += reward - lift_penalty - robot_penalty + lift_bonus
     
        self.reward += reward
        if np.sum(self.cargo_map) == 0:
            self.reward += self.max_steps - self.current_step  # 스텝 수에 따른 보상 감소
            done = True
            
            ## 출력 2가지 스타일
            # if self.episode % 100 == 0:
                # print(f"Episode {self.episode}. Steps taken: {self.current_step}. Remaining soil: {np.sum(self.cargo_map)}. Total reward: {self.reward}")
            print(f"Episode {self.episode}. Steps taken: {self.current_step}. Remaining soil: {np.sum(self.cargo_map)}. Total reward: {self.reward}")
            self.episode += 1
        else:
            done = False

        #print("np.sum(self.cargo_map)::",np.sum(self.cargo_map))
        self.current_step += 1
        if self.current_step >= self.max_steps:
            print(f"Episode {self.episode}. Steps taken: {self.current_step}. Remaining soil: {np.sum(self.cargo_map)}. Total reward: {self.reward}")
            done = True
            self.episode += 1


        if self.graphic == True and self.render_mode == "human":
            self.render_frame()
        return self._next_observation(), self.reward, done, {}

    def render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.win_size, self.win_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.win_size, self.win_size))
        canvas.fill((255, 255, 255))

        # 수화물 크기
        cargo_pix_size = 9
        # workspace의 구분선 크기
        line_width = 3

        for y in range(self.map_size):
            for x in range(self.map_size):
                n_soil = self.cargo_map[y][x]
                for z in range(n_soil):
                    # Calculate the top-left corner of the soil cell
                    top_left_x = x * self.cell_size + z * cargo_pix_size + line_width
                    top_left_y = y * self.cell_size + line_width
                    # Draw the soil cell
                    pygame.draw.rect(canvas, (146, 104, 41), pygame.Rect(top_left_x, top_left_y, cargo_pix_size, cargo_pix_size))
                    # Draw the grid lines around the soil cell
                    pygame.draw.rect(canvas, (0, 0, 0), pygame.Rect(top_left_x - line_width, top_left_y - line_width, cargo_pix_size + line_width, cargo_pix_size + line_width), width=line_width)


        pygame.font.init()
        font = pygame.font.SysFont(None, 25)  # 기본 폰트와 크기 25로 설정

        lift_pix_size = 30
        for i in range(self.num_lift):
            rows, cols = self.lift_positions[i][1], self.lift_positions[i][0]
            color = (0, 255, 0) if i == 1 else (0, 0, 255)
            carry_value = self.lift_carry[i]
            lift_y_offset = self.cell_size / 3
            lift_x_offset = (i % 3) * (self.cell_size / 3)
            carry_text = font.render(str(carry_value), True, (255, 255, 255))
            text_x = cols*self.cell_size + lift_x_offset + lift_pix_size/2 - carry_text.get_width() / 2
            text_y = rows*self.cell_size + lift_y_offset + lift_pix_size/2 - carry_text.get_height() / 2
            excavator_rect = pygame.Rect(
                cols*self.cell_size + lift_x_offset,
                rows*self.cell_size + lift_y_offset,
                lift_pix_size,
                lift_pix_size
            )
            pygame.draw.rect(canvas, color, excavator_rect)
            canvas.blit(carry_text, (text_x, text_y))


        
        lift_pix_size = 30
        for i in range(self.num_robot):
            rows, cols = self.robot_positions[i][1], self.robot_positions[i][0]
            color = (0, 0, 0)
            
            robot_y_offset = (self.cell_size / 3) * 2
            robot_x_offset = (i % 3) * (self.cell_size / 3)
            load_text = font.render(str(self.robot_load[i]), True, (255, 255, 255))
            # 글자 위치도 덤프트럭의 x 오프셋을 고려하여 조정합니다.
            text_x = cols*self.cell_size + robot_x_offset + lift_pix_size/2 - load_text.get_width()/2
            text_y = rows*self.cell_size + robot_y_offset + lift_pix_size/2 - load_text.get_height()/2
            # 덤프트럭을 그립니다.
            robot_rect = pygame.Rect(
                cols*self.cell_size + robot_x_offset,
                rows*self.cell_size + robot_y_offset,
                lift_pix_size,
                lift_pix_size
            )
            pygame.draw.rect(canvas, color, robot_rect)
            
            canvas.blit(load_text, (text_x, text_y))

        for x in range(self.map_size + 1):
            pygame.draw.line(canvas, (0, 0, 0), (0, self.cell_size*x), (self.win_size, self.cell_size*x), width=line_width)
            pygame.draw.line(canvas, (0, 0, 0), (self.cell_size*x, 0), (self.cell_size*x, self.win_size), width=line_width)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.draw_fps)
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


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
