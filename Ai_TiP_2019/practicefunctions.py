#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 11:52:10 2019

@author: rmoctezuma
"""
# PYTHON PRACTICE PROBLEMS - FUNCTIONS

import string

def hello_world():
    print('hello world')

#1
def to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5/9
def to_fahrenheit(celsius):
    return (celsius * 9/5) + 32
#2
def triple_sum(a,b,c):
    return a + b + c
#3
def gauss(n):
    return (n/2)*(n+1)
#4
def maximum(a,b,c):
    highest_number = a
    if a > b and a > c:
        highest_number = a
    elif b > a and b > c:
        highest_number = b
    elif c > a and c > b:
        highest_number = c
    return highest_number
#5
def triangle(a,b,c):
    if a == b == c:
        return 'equilateral'
    elif (a == b and not a == c) or (b == c and not b == a) or (c == a and not c == b):
        return 'isosceles'
    else:
        return 'scalene'
#6
def median(a,b,c):
    s = [a,b,c]
    m = s.index(maximum(a,b,c))
    x = s[2]
    s[2] = maximum(a,b,c)
    s[m] = x
    if s[0] > s[1]:
        return s[0]
    else:
        return s[1]
#7
def add(a,b):
    return a + b
#8
def avg(a):
    s = 0.0
    for val in a:
        s = add(s,val)
        print(val)
    return s/len(a)
#9
def perfect_squares(n):
    a = []
    val = 0
    while val <= n:
        a.append((val**2))
        val += 1
    return a
#12
def in_function(x, y):
    b = False
    for a in y:
        if a == x:
            b = True
    return b
#11
def odd_nums(a):
    count = 0
    for x in a:
        if x % 2 == 1:
            count += 1
    return count
#10
def charcount(s):
    digit_count = 0
    letter_count = 0
    for val in s:
        if in_function(val, string.ascii_lowercase) or in_function(val, string.ascii_uppercase):
            letter_count += 1
        elif in_function(val, ['0','1','2','3','4','5','6','7','8','9']):
            digit_count += 1
    return(digit_count,letter_count)
#13
def password(p):
    if len(p) < 8 or len(p) > 32:
        return "Invalid"
    for l in p:
        if in_function(l, string.ascii_lowercase):
            break
        elif l == p[len(p) - 1]:
            return "Invalid"
    for l in p:
        if in_function(l, string.ascii_uppercase):
            break
        elif l == p[len(p) - 1]:
            return "Invalid"
    for l in p:
        if in_function(l, ['0','1','2','3','4','5','6','7','8','9']):
            break
        elif l == p[len(p) - 1]:
            return "Invalid"
    return "Valid"
#14
def reverse_function(s):
    r = ""
    for val in range(len(s) - 1,-1,-1):
        r += s[val]
    return r
#15
def palindrome(s):
    if s == reverse_function(s):
        return True
    else:
        return False
#16
def pangram(s):
    l = 'a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'
    l = l.split(',')
    for val in s:
        if in_function(val, l):        
            l.remove(val)
            l.remove(val.swapcase())
    print(l)
    if l == []:
        return 'This is a pangram'
    return 'This is not a pangram'
#17
def unique(s):
    a = 0
    for val in s:
        if a != len(s) - 1:
            x = a + 1
            while x < len(s):
                if val == s[x]:
                    del s[x]
                else:
                    x += 1
        a += 1
    return s
#18
def is_prime(p):
    if p == 0:
        return '0 is not a prime number.'
    x = 2 
    while x <= p/2:
        if (p/x)%1 == 0:
            return str(p) + ' is not a prime number.'
        x += 1
    return str(p) + ' is a prime number.'
#19 
def up_to_prime(n):
    primes = []
    for val in range(0,n + 1,1):
        if is_prime(val) == str(val) + ' is a prime number.':
            primes.append(val)
    return primes
#20
def perfect_number(n):
    divisor_sum = 0
    for val in range(1,n,1):
        if (n/val)%1 == 0:
            divisor_sum += val
    if divisor_sum == n:
        return str(n) + ' is a perfect number.'
    else:
        return str(n) + ' is not a perfect number.'
#21
def multiplication_table(n):
    a = []
    b = []
    for val in range(1,n+1,1):
        for x in range(1,n+1,1):
            b.append(val*x)
        a.append(b)
        b = []
    return a
#22
def pyramid(n):
    for val in range(1,n + 1,1):
        print(str(val)*val)
#23
def factorial(n):
    if n == 0:
        return 1
    return n*factorial(n-1)
#24
def fibonacci(n):
    if n <= 1:
        return n
    else:
       return(fibonacci(n-1) + fibonacci(n-2))
#25
def pascal(n, a = []):
    
    if len(a) == 0:
        return 1
    if n == 1:
        return a[0] + [1]
    else:
        return a[0] + ((factorial(n)/(factorial(pascal(n-1, a[1:]))*factorial(n-pascal(n-1, a[1:])))))

  """  a = []
    b = []
    for val in range(1,n+1,1):
        for x in range(0,val+2,1):
            if x  == 0:
                a.append(0)
            elif x == 1:
                a.append(1)
            elif x == val:
                a.append(1)
            elif x == val + 1:
                a.append(0)
            else:
                a.append((b[x]) + (b[x-1]))
        b = a
        a = []
        
    return b"""
#   CLASSES
#27
class Shape(object):
    def __init__(self,a,p,v):
        self.area = a
        self.perimeter = p
        self.vertices = v
    def perimeter_area_ratio(self):
        return self.perimeter/self.area
    def diagonals(self):
        return (self.vertices*(self.vertices-3))/2
        
class Circle(Shape):
    def __init__(self,a,p):
        Shape.__init__(self,a,p,0)

class Rectangle(Shape):
    def __init__(self,a,p):
        Shape.__init__(self,a,p,4)
        
class Square(Rectangle):
    def __init__(self,a,p):
        Rectangle.__init__(self,a,p)
class Regular_Polygon(Shape):
    def __init(self,a,p,v):
        Shape.__init__(self,a,p,v)