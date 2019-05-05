#!/usr/local/bin/python3
# -*- coding: utf-8 -*-

import sys
import time
import threading
import math

import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtCore import pyqtSignal, QSize, Qt, QTimer

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class GLmodel(QtWidgets.QOpenGLWidget):
    def __init__(self, parent=None, aim=None):
        super(GLmodel, self).__init__(parent)
        self.object = []
        self.objects = []
        self.zoom = -30.0
        self.xRot = 0
        self.yRot = 0
        self.zRot = 0
        self.w = 0
        self.h = 0
        self.v = 0
        self.a = 0.1
        self.s = 8
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.dxRot = 0
        self.dyRot = 0
        self.dzRot = 0
        self.e = 1

        self.aim = aim
        self.pointsize = 3

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)

        lightSpecular =  (1.0, 1.0, 1.0, 1.0)
        lightPosition =  (0.5, 0.5, 4.0, 0.0)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glShadeModel(GL_SMOOTH)

        glMaterialfv(GL_FRONT, GL_SPECULAR, lightSpecular)
        glMaterialf(GL_FRONT, GL_SHININESS, 80.0)
        glLightfv(GL_LIGHT0, GL_POSITION, lightPosition)

        glEnable(GL_BLEND)
        # glEnable(GL_LIGHTING)
        # glEnable(GL_LIGHT0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_COLOR_MATERIAL)
    
    # def resizeGL(self, width, height):
    #     aspect = width/height
    #     self.w = width
    #     self.h = height

    #     glViewport(0, 0, width, height)
    #     self.v = glGetIntegerv(GL_VIEWPORT)
    #     glMatrixMode(GL_PROJECTION)
    #     glLoadIdentity()
    #     gluPerspective(35.0, aspect, 1.0, 1000.0)
    #     glMatrixMode(GL_MODELVIEW)
    #     glLoadIdentity()
    
    # def paintGL(self):
    #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    #     glColor3f(1.0, 1.0, 1.0)
    #     glLoadIdentity()             # clear the matrix
    #     # # draw background
    #     # glPushMatrix()
    #     glTranslatef(0, 0, -1700)
    #     glBegin(GL_TRIANGLE_FAN)
    #     glColor3f(0.0, 0.0, 1.0)
    #     glVertex3f(1000.0, 1000.0, 0.0)
    #     glColor3f(0.0, 0.0, 0.0)
    #     glVertex3f(1000.0, -1000.0, 0.0)
    #     glColor3f(0.0, 0.0, 0.0)
    #     glVertex3f(-1000.0, -1000.0, 0.0)
    #     glColor3f(0.0, 0.0, 1.0)
    #     glVertex3f(-1000.0, 1000.0, 0.0)
    #     glEnd()

    #     # glPopMatrix()
    #     glLoadIdentity()             # clear the matrix
    #     glColor3f(1.0, 1.0, 1.0)
    #     # viewing transformation
    #     gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    #     glTranslatef(0.0, 0.0, -10.0)
    #     glRotatef(self.xRot, 1.0, 0.0, 0.0)
    #     glRotatef(self.yRot, 0.0, 1.0, 0.0)
    #     glRotatef(self.zRot, 0.0, 0.0, 1.0)

    #     glScalef(1.0, 2.0, 1.0)      # modeling transformation

    #     # glutWireCube(1.0)
    #     glFlush()