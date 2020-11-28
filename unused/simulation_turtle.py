#https://github.com/DCoelhoM/Snake-Python/blob/master/Snake_by_DCM.py

import turtle
import os
import random
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using :0.0')
    os.environ.__setitem__('DISPLAY', ':0.0')

def home():

    turtle.hideturtle()
    turtle.clear()
    turtle.pu()
    turtle.color("black")
    turtle.goto(0,0)
    turtle.write("Start")
    turtle.title("Thymio simulation")
    turtle.onscreenclick(start)
    turtle.mainloop()

def map():
    turtle.clear()
    turtle.pu()
    turtle.speed(0)
    turtle.pensize(20)
    turtle.color("grey")
    turtle.goto(-220,220)
    turtle.pd()
    turtle.goto(220,220)
    turtle.goto(220,-220)
    turtle.goto(-220,-220)
    turtle.goto(-220,220)
    turtle.pu()
    turtle.goto(0,0)

def start(x,y):
    turtle.onscreenclick(None)
    map()

    turtle_th.pu()
    turtle_th.speed(0)
    turtle_th.shape("square")
    turtle_th.color("red")

    turtle_move(25,5)


def turtle_move(x,y):
    x_prec = turtle_th.xcor()
    y_prec = turtle_th.ycor() 
        
    turtle_th.goto(x,y)
    print(x,y)

if __name__ == '__main__':
    turtle_th = turtle.Turtle()
    home()

    