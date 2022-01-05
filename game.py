# 장애물 회피 게임 즉, 자율주행차:-D 게임을 구현합니다.
import numpy as np
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Game:
    def __init__(self, screen_width, screen_height, show_game=True):
        self.screen_width = screen_width
        self.screen_height = screen_height

        # 자동차와 장애물의 초기 위치와, 장애물 각각의 속도를 정합니다.
        self.enemy = {"col": 0, "row": 0}
        self.local = [
            {"col": 2, "row": 2},
            {"col": 5, "row": 2},
            {"col": 6, "row": 0},
            {"col": 9, "row": 2},
            {"col": 13, "row": 1},
            {"col": 11, "row": 3},
            {"col": 3, "row": 4},
            {"col": 1, "row": 5},
            {"col": 7, "row": 4},
            {"col": 12, "row": 5},
            {"col": 3, "row": 7},
            {"col": 9, "row": 7},
            {"col": 13, "row": 7},
            {"col": 14, "row": 4},
            {"col": 0, "row": 8},
            {"col": 7, "row": 8},
            {"col": 11, "row": 9},
            {"col": 1, "row": 10},
            {"col": 7, "row": 10},
            {"col": 13, "row": 10},
            {"col": 6, "row": 12},
            {"col": 9, "row": 12},
            {"col": 13, "row": 12},
            {"col": 3, "row": 13},
            {"col": 10, "row": 14}
        ]
        self.car = []
        self.car_pos1 = 0
        self.car_pos2 = 0

        self.road1 = [
            {"col": 2, "row": 3},
            {"col": 2, "row": 4},
            {"col": 2, "row": 5},
            {"col": 2, "row": 6},
            {"col": 2, "row": 7},
            {"col": 2, "row": 8},
            {"col": 2, "row": 9},
            {"col": 2, "row": 10},
            {"col": 2, "row": 11},
            {"col": 3, "row": 11},
            {"col": 4, "row": 11},
            {"col": 5, "row": 11},
            {"col": 6, "row": 11},
            {"col": 7, "row": 11},
            {"col": 8, "row": 11},
            {"col": 8, "row": 10},
            {"col": 8, "row": 9},
            {"col": 8, "row": 8},
            {"col": 8, "row": 7},
            {"col": 8, "row": 6},
            {"col": 8, "row": 5},
            {"col": 8, "row": 4},
            {"col": 8, "row": 3},
            {"col": 7, "row": 3},
            {"col": 6, "row": 3},
            {"col": 5, "row": 3},
            {"col": 4, "row": 3},
            {"col": 3, "row": 3}
        ]
        self.road2 = [
            {"col": 8, "row": 6},
            {"col": 8, "row": 7},
            {"col": 8, "row": 8},
            {"col": 8, "row": 9},
            {"col": 8, "row": 10},
            {"col": 8, "row": 11},
            {"col": 8, "row": 12},
            {"col": 8, "row": 13},
            {"col": 9, "row": 13},
            {"col": 10, "row": 13},
            {"col": 11, "row": 13},
            {"col": 12, "row": 13},
            {"col": 12, "row": 12},
            {"col": 12, "row": 11},
            {"col": 12, "row": 10},
            {"col": 12, "row": 9},
            {"col": 12, "row": 8},
            {"col": 12, "row": 7},
            {"col": 12, "row": 6},
            {"col": 11, "row": 6},
            {"col": 10, "row": 6},
            {"col": 9, "row": 6}
        ]
        self.obj = [
            {"col": 4, "row": 10},
            {"col": 11, "row": 12}
        ]
        self.dist_obj = []
        temp = [0, 0]
        for i in range(len(self.obj)):
            temp[i] = abs(self.enemy["col"] - self.obj[i]["col"]) + abs(self.enemy["row"] - self.obj[i]["row"])
        self.dist_obj.append(temp)
        self.obj_break = [False, False]

        self.total_reward = 0.
        self.current_reward = 0.
        self.total_game = 0
        self.show_game = show_game

        if show_game:
            self.fig, self.axes = self._prepare_display()

    def _prepare_display(self):
        """게임을 화면에 보여주기 위해 matplotlib 으로 출력할 화면을 설정합니다."""
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        fig.set_size_inches(self.screen_width + 2, self.screen_height + 2)
        # 화면을 닫으면 프로그램을 종료합니다.

        plt.axis((-1, self.screen_width + 1, -1, self.screen_height + 1))
        plt.tick_params(top='off', right='off',
                        left='off', labelleft='off',
                        bottom='off', labelbottom='off')

        plt.draw()
        # 게임을 진행하며 화면을 업데이트 할 수 있도록 interactive 모드로 설정합니다.
        plt.ion()
        plt.show()

        return fig, axes

    def _get_state(self):
        """게임의 상태를 가져옵니다.
        게임의 상태는 screen_width x screen_height 크기로 각 위치에 대한 상태값을 가지고 있으며,
        빈 공간인 경우에는 0, 특작군인 경우에는 1, 지역방어대인 경우에는 2, 기동전력인 경우에는 3, 보상의 경우에는 4가 들어있는 2차원 배열입니다.
        """
        state = np.zeros((self.screen_width + 2, self.screen_height + 2))

        for i in range(len(self.local)):
            state[self.local[i]["col"]+1, self.local[i]["row"]+1] = 2

        for i in range(len(self.car)):
            state[self.car[i]["col"]+1, self.car[i]["row"]+1] = 3

        for i in range(len(self.obj)):
            state[self.obj[i]["col"]+1, self.obj[i]["row"]+1] = 4

        state[self.enemy["col"]+1, self.enemy["row"]+1] = 1

        return state

    def _draw_screen(self):
        plt.cla()
        title = " Avg. Reward: %d Reward: %d Total Game: %d" % (
                        self.total_reward / self.total_game,
                        self.current_reward,
                        self.total_game)
        # self.axis.clear()
        self.axes.set_title(title, fontsize=20)

        for i in range(-1, self.screen_width + 2):
            self.axes.plot([i, i], [-1, self.screen_height], color='black', linewidth=1)
        for i in range(-1, self.screen_height + 2):
            self.axes.plot([-1, self.screen_width], [i, i], color='black', linewidth=1)

        for i in range(len(self.local)):
            self.axes.scatter(self.local[i]["col"], self.local[i]["row"], marker='x', color='green')

        for i in range(len(self.obj)):
            self.axes.scatter(self.obj[i]["col"], self.obj[i]["row"], marker='^', color='magenta')

        for i in range(len(self.car)):

            self.axes.scatter(self.car[i]["col"], self.car[i]["row"], marker='s', color='blue')

        self.axes.scatter(self.enemy["col"], self.enemy["row"], marker='o', color='red')

        self.fig.canvas.draw()
        # 게임의 다음 단계 진행을 위해 matplotlib 의 이벤트 루프를 잠시 멈춥니다.
        plt.pause(0.001)

    def reset(self):
        # 특작군, 기동전력,
        self.current_reward = 0
        self.total_game += 1

        self.enemy["col"] = 0
        self.enemy["row"] = 0

        self.car.clear()
        self.car_pos1 = random.randrange(0, len(self.road1))
        self.car_pos2 = random.randrange(0, len(self.road2))
        self.car.append(self.road1[self.car_pos1])
        self.car.append(self.road2[self.car_pos2])
        self.obj_break = [False, False]

        return self._get_state()

    def _update_enemy(self, move):
        # 액션에 따라 특작군을 이동시킵니다.
        # action 0: 상, 1: 우
        if move == 0:
            self.enemy["row"] = self.enemy["row"] + 1
        elif move == 1:
            self.enemy["col"] = self.enemy["col"] + 1

    def _update_env(self):

        reward = 0
        # 기동전력의 이동(순찰)
        self.car_pos1 += 1
        if self.car_pos1 >= len(self.road1):
            self.car_pos1 = 0

        self.car_pos2 += 1
        if self.car_pos2 >= len(self.road2):
            self.car_pos2 = 0

        self.car.clear()
        self.car.append(self.road1[self.car_pos1])
        self.car.append(self.road2[self.car_pos2])

        temp = [0, 0]
        for i in range(len(self.obj)):
            temp[i] = abs(self.enemy["col"] - self.obj[i]["col"]) + abs(self.enemy["row"] - self.obj[i]["row"])
        self.dist_obj.append(temp)
        # 파괴에 성공했을 때의 보상
        for i in range(len(self.obj)):
            if temp[i] == 0 and not self.obj_break[i]:
                reward += 100
                self.obj_break[i] = True

        return reward

    def _is_gameover(self):
        # 캐릭터가 죽었다면 dead가 True, 살았다면 dead가 False
        dead = False
        point = 0
        # 특작군이 기동전력에 식별 되었을때
        for i in range(len(self.car)):
            if abs(self.enemy["col"]-self.car[i]["col"])+abs(self.enemy["row"]-self.car[i]["row"]) <= 1:
                dead = dead or True
                point = -20
        # 특작군이 지역방어대에 식별되었을때
        for i in range(len(self.local)):
            if abs(self.enemy["col"]-self.local[i]["col"])+abs(self.enemy["row"]-self.local[i]["row"]) == 0:
                dead = dead or True
                point = -20
        # 특작군이 grid 밖으로 나갔을 때
        if self.enemy["col"] < 0 or \
                self.enemy["col"] >= self.screen_width or \
                self.enemy["row"] < 0 or \
                self.enemy["row"] >= self.screen_height:
            dead = dead or True
            point = -100

        return point, dead

    def step(self, action):
        # action: 0: 상, 1: 우
        # 특작군을 DQN 에서 얻은 action 으로 1스텝 업데이트한다.
        self._update_enemy(action)
        # 환경을 1스텝 업데이트한다.
        break_reward = self._update_env()
        # 게임이 종료됐는지를 판단
        death_reward, gameover = self._is_gameover()

        if gameover:
            reward = death_reward
            self.total_reward += (self.current_reward + reward)
        else:
            reward = break_reward
            self.current_reward += reward

        temp = True
        for i in range(len(self.obj_break)):
            temp = temp and self.obj_break[i]

        gameover = gameover or temp

        if gameover and temp:
            self.total_reward += self.current_reward

        if self.show_game:
            self._draw_screen()

        return self._get_state(), reward, gameover