import numpy as np
from pyquaternion import Quaternion
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from time import sleep
import math

animation_parameter = 0
animation_parameter_max = 400
animation_ongoing = 0


def Euler2A(phi, theta, psi):
  Rx_phi = [[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]]
  Ry_theta = [[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]]
  Rz_psi = [[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]]
  A = np.matmul(Rz_psi, np.matmul(Ry_theta, Rx_phi))
  return A

def AxisAngle(A):
    det = np.linalg.det(A)
    if det < 0.99999999 or det > 1.00000001:
        print("Matrica prosledjena f-ji AxisAngle nema determinantu 1, vec:")
        print(det)
        sys.exit(1)
        
    A1 = A - np.eye(3)
    p = np.cross(A1[0], A1[1])
    p = p / np.linalg.norm(p)
    u = A1[0]
    u = u / np.linalg.norm(u)
    u_prime = A.dot(u)
    u_prime = u_prime / np.linalg.norm(u_prime)
    phi = np.arccos(u.dot(u_prime))
    if np.linalg.det(np.array([u, u_prime, p])) < 0:
        p = -p
    return p, phi


def Rodrigez(p, fi):
  p_intensity = math.sqrt(np.dot(p, p))
  p_unit = np.array([p[0] / p_intensity, p[1] / p_intensity, p[2] / p_intensity])

  pT = np.transpose([p_unit])
  ppT = np.matmul(pT, [p_unit])
  identity = np.eye(3, dtype=float)
  pX = np.array([[0, -p_unit[2], p_unit[1]], [p_unit[2], 0, -p_unit[0]], [-p_unit[1], p_unit[0], 0]])

  A = ppT + math.cos(fi) * (identity - ppT) + math.sin(fi) * pX

  return A


def A2Euler(A):
  det = np.linalg.det(A)
  if det < 0.99999999 or det > 1.00000001:
    print("Matrica prosledjena f-ji A2Euler nema determinantu 1, vec:")
    print(det)
    sys.exit(1)

  psi = -100
  theta = -100
  fi = -100

  if A[2, 0] < 1:
    if A[2,0] > -1:
      psi = np.arctan2(A[1,0], A[0, 0])
      theta = math.asin(-A[2,0])
      fi = np.arctan2(A[2, 1], A[2, 2])
    else:
      psi = np.arctan2(-A[0, 1], A[1, 1])
      theta = math.pi / 2
      fi = 0
  else:
    psi = np.arctan2(-A[0, 1], A[1, 1])
    theta = -math.pi / 2
    fi = 0

  return fi, theta, psi

def AxisAngle2Q(p, phi):
    w = np.cos(phi / 2)
    p = p / np.linalg.norm(p)
    im = np.sin(phi / 2) * p
    q = Quaternion(imaginary=im, real=w)
    return q


def Q2AxisAngle(q):
    q = q.normalised
    if q.real < 0:
        q = -q
    phi = 2 * np.arccos(q.real)
    if q.real == 1:
        p = np.eye(1, 3)
    else:
        p = q.imaginary
        p = p / np.linalg.norm(p)

    return p, phi

def slerp(q1, q2, t, tm):
    dot = q1.conjugate.real * q2.real
    if dot < 0:
        q1 = -q1
        dot = -dot
    if dot > 0.95:
        return q1

    phi = np.arccos(dot)
    qs = (np.sin(phi * (1 - t / tm)) / np.sin(phi)) * q1 + (np.sin(phi * (t / tm)) / np.sin(phi)) * q2
    return qs


def glut_initialization():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(1366, 768);
    glutCreateWindow("SLERP")

    glClearColor(0, 0, 0, 1)
    glEnable(GL_DEPTH_TEST)

    glutDisplayFunc(on_display)
    glutKeyboardFunc(on_keyboard)
    glutIdleFunc(on_animate)

    glEnable(GL_NORMALIZE)
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glLineWidth(2)


def glut_perspective():
    glMatrixMode(GL_PROJECTION)
    gluPerspective(50, 1, 1, 50)
    glMatrixMode(GL_MODELVIEW)
    gluLookAt(6, 5, 12.2,
              0, 0, 0,
              0, 1, 0)
    glPushMatrix()
    glutMainLoop()

def on_animate():
    global animation_parameter, animation_parameter_max, animation_ongoing

    if animation_ongoing == 1:    
        if animation_parameter == animation_parameter_max:
            sleep(1)
            animation_parameter = 0
        animation_parameter += 1
    glutPostRedisplay()


def on_keyboard(key, x, y):
    global animation_ongoing
    if key == b'q':
        exit()
    elif key == b'g' and animation_ongoing == 0:
        animation_ongoing = 1
    elif key == b's' and animation_ongoing == 1:
        animation_ongoing = 0


def material(i):
    if i == 0:
        glMaterialfv(GL_FRONT, GL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 0.2))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, GLfloat_4(0.2, 0.8, 0.2, 0.2))
        glMaterialfv(GL_FRONT, GL_SPECULAR, GLfloat_4(0.4, 1.0, 0.4, 0.2))
        glMaterialfv(GL_FRONT, GL_SHININESS, GLfloat(40.0))       
    elif i == 2:
        glMaterialfv(GL_FRONT, GL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, GLfloat_4(0.2, 0.8, 0.2, 1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR, GLfloat_4(0.4, 1.0, 0.4, 1.0))
        glMaterialfv(GL_FRONT, GL_SHININESS, GLfloat(40.0))
    elif i == 3:
        glMaterialfv(GL_FRONT, GL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, GLfloat_4(0.2, 0.2, 0.8, 1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR, GLfloat_4(0.4, 0.4, 1.0, 1.0))
        glMaterialfv(GL_FRONT, GL_SHININESS, GLfloat(40.0))
    elif i == 1:       
        glMaterialfv(GL_FRONT, GL_AMBIENT, GLfloat_4(0.2, 0.2, 0.2, 1.0))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, GLfloat_4(0.8, 0.2, 0.2, 1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR, GLfloat_4(1.0, 0.4, 0.4, 1.0))
        glMaterialfv(GL_FRONT, GL_SHININESS, GLfloat(40.0))


def lighting():
    glLightfv(GL_LIGHT0, GL_AMBIENT, GLfloat_4(0.1, 0.1, 0.1, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, GLfloat_4(0.8, 0.8, 0.8, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, GLfloat_4(1.0, 1.0, 1.0, 1.0))
    glLightfv(GL_LIGHT0, GL_POSITION, GLfloat_4(-1.0, 8.0, 2.0, 0.0))
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)


def draw_axes():
    material(1)
    
    glPushMatrix()
    glScalef(2.0, 0.1, 0.1)
    glTranslatef(0.5, 0, 0)
    glutSolidCube(1)
    glPopMatrix()
    
    material(2)

    glPushMatrix()

    glScalef(0.1, 2.0, 0.1)
    glTranslatef(0, 0.5, 0)
    glutSolidCube(1)
    glPopMatrix()

    material(3)

    glPushMatrix()
    glScalef(0.1, 0.1, 2.0)
    glTranslatef(0, 0, 0.5)
    glutSolidCube(1)
    glPopMatrix()


def on_display():
    global animation_parameter, animation_parameter_max
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    point = np.array([4, -2, -3])
    euler_angles = np.array([19 * np.pi / 7, 3 * np.pi / 7, 4 * np.pi / 5])
    A = Euler2A(euler_angles[0], euler_angles[1], euler_angles[2])
    p, phi = AxisAngle(A)
    q = AxisAngle2Q(p, phi)

    point2 = np.array([-4, 2, 3])
    euler_angles2 = np.array([np.pi / 11, -2 * np.pi / 3, 15 * np.pi / 19])
    A2 = Euler2A(euler_angles2[0], euler_angles2[1], euler_angles2[2])
    p2, phi2 = AxisAngle(A2)
    q2 = AxisAngle2Q(p2, phi2)

    qs = slerp(q, q2, animation_parameter, animation_parameter_max)
    
    p_s, phi_s = Q2AxisAngle(qs) 
    A_s = Rodrigez(p_s, phi_s)
    phi, theta, psi = A2Euler(A_s)
    c = (1 - animation_parameter / animation_parameter_max) * point + (animation_parameter / animation_parameter_max) * point2

    lighting()

    glPushMatrix()
    glTranslatef(point[0], point[1], point[2])
    glRotatef(np.rad2deg(euler_angles[0]), 1, 0, 0)
    glRotatef(np.rad2deg(euler_angles[1]), 0, 1, 0)
    glRotatef(np.rad2deg(euler_angles[2]), 0, 0, 1)
    draw_axes()
    material(1)
    glutWireIcosahedron()
    glPopMatrix()    
    

    glPushMatrix()
    glTranslatef(c[0], c[1], c[2])
    glRotatef(np.rad2deg(phi), 1, 0, 0)
    glRotatef(np.rad2deg(theta), 0, 1, 0)
    glRotatef(np.rad2deg(psi), 0, 0, 1)
    draw_axes()
    material(0)
    glutSolidIcosahedron()
    glPopMatrix() 

    glPushMatrix()
    glTranslatef(point2[0], point2[1], point2[2])
    glRotatef(np.rad2deg(euler_angles2[0]), 1, 0, 0)
    glRotatef(np.rad2deg(euler_angles2[1]), 0, 1, 0)
    glRotatef(np.rad2deg(euler_angles2[2]), 0, 0, 1)
    draw_axes()
    material(3)
    glutWireIcosahedron()
    glPopMatrix()

    draw_axes()

    glDisable(GL_LIGHTING)
    glDisable(GL_LIGHT0)

    glutSwapBuffers()

def main():

    glut_initialization()
    glut_perspective()


if __name__ == '__main__':
    main()

