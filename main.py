"""_summary_
文字認識アプリケーション
"""

import sys
import numpy as np
import pygame
from pygame.locals import *
from layers.two_layer_net_backprop import TwoLayerNet

# init
pygame.init()
screen = pygame.display.set_mode((600, 400))
gray = (122,122,122)
black = (0, 0, 0)

# init DNN
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10, load=True)


# cellのx、yの数
input_x = 28
input_y = 28

# Init canvas
input_field = [[(0,0,0)]*input_x for i in range(input_y)]

cell_size = 10      # cell size
offset_x = 20       # offset canvas x
offset_y = 50       # offset canvas y
isPressed = False

# Init Reset button
button_x = 100
button = pygame.Rect(offset_x+((cell_size*input_x)/2)-button_x/2, offset_y+cell_size*input_y+10, button_x, 50)

# font setting
font = pygame.font.SysFont(None, 25)

# text
text1 = font.render("Reset", True, (0,0,0))
text2 = font.render("Prediction: ", True, (0,0,0))

cell = [0 for x in range(input_x*input_y)]

predict = " "

while True:
    screen.fill(gray)
    pygame.draw.rect(screen, (200, 200, 200), button)
    screen.blit(text1, (offset_x+((cell_size*input_x)/2)-25, offset_y+cell_size*input_y+25))
    screen.blit(text2, (offset_x+cell_size*input_x+50, 80))

    # prediction text
    font = pygame.font.SysFont(None, 200)
    prediction = font.render(predict, True, (0,0,0))
    screen.blit(prediction, (offset_x+cell_size*input_x+100, 120))

    index = 0
    # Update Canvas Info and Convert canvas list to numpy array
    for i in range(input_x):
        for j in range(input_y):
            pygame.draw.rect(screen, input_field[i][j], (i*cell_size+offset_x, j*cell_size+offset_y, cell_size, cell_size), )

            cell[index] = np.mean(input_field[j][i])
            index+=1
    np_cell = np.array(cell)/255.0

    # Update Display
    pygame.display.update()

    for event in pygame.event.get():
        # Drawing Process
        if event.type ==MOUSEMOTION and isPressed:
                mouse_Pos = pygame.mouse.get_pos()
                if mouse_Pos[0] >= cell_size*input_x+offset_x or mouse_Pos[1] >= cell_size*input_y + offset_y or mouse_Pos[0] < offset_x or mouse_Pos[1] < offset_y:
                    continue

                # マウスがいるセルを計算
                x = int((mouse_Pos[0]-offset_x)/cell_size)
                y = int((mouse_Pos[1]-offset_y)/cell_size)

                # マウスがあるセルを真っ白に、上下左右は+50明るくする
                input_field[x][y] = (230,230,230)
                if x+1<input_x:
                    lst = np.array(input_field[x+1][y])
                    lst += 50
                    lst = np.clip(lst, 0, 255)
                    input_field[x+1][y] = tuple(lst)
                if y+1 < input_y:
                    lst = np.array(input_field[x][y+1])
                    lst += 50
                    lst = np.clip(lst, 0, 255)
                    input_field[x][y+1] = tuple(lst)
                if x-1 >= 0:
                    lst = np.array(input_field[x-1][y])
                    lst += 50
                    lst = np.clip(lst, 0, 255)
                    input_field[x-1][y] = tuple(lst)
                if y-1 >= 0:
                    lst = np.array(input_field[x][y-1])
                    lst += 50
                    lst = np.clip(lst, 0, 255)
                    input_field[x][y-1] = tuple(lst)

                # Prediction
                predict = str(np.argmax(network.predict(np_cell)))

        if event.type == MOUSEBUTTONDOWN:
            if event.button == 1:
                # flag
                isPressed = True

            # Reset Button Process
            if button.collidepoint(event.pos):
                # clear canvas
                input_field = [[(0,0,0)]*input_x for i in range(input_y)]
                predict = " "

        elif event.type == MOUSEBUTTONUP:
            # flag
            isPressed = False

        if event.type == QUIT:
            pygame.quit()
            sys.exit()